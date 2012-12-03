/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011-2012  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
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

/* Distributed queues using performance modeling to assign tasks */

#include <float.h>

#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>
#include <core/task_bundle.h>
#include <core/workers.h>
#include <starpu_parameters.h>
#include <starpu_task_bundle.h>
#include <starpu_top.h>
#include <core/jobs.h>
#include <top/starpu_top_core.h>
#include <sched_policies/fifo_queues.h>
#include <core/debug.h>

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif


static double current_time[STARPU_NMAXWORKERS][STARPU_NMAX_SCHED_CTXS];

typedef struct {
	double alpha;
	double beta;
	double _gamma;
	double idle_power;
	struct _starpu_fifo_taskq **queue_array;
} heft_data;

const float alpha_minimum=0;
const float alpha_maximum=10.0;
const float beta_minimum=0;
const float beta_maximum=10.0;
const float gamma_minimum=0;
const float gamma_maximum=10000.0;
const float idle_power_minimum=0;
const float idle_power_maximum=10000.0;

static void param_modified(struct starpu_top_param* d)
{
	//just to show parameter modification
	fprintf(stderr,"%s has been modified : %f !\n", d->name, *(double*)d->value);
}


static void heft_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	int workerid;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		hd->queue_array[workerid] = _starpu_create_fifo();
		starpu_worker_init_sched_condition(sched_ctx_id, workerid);

		current_time[workerid][sched_ctx_id] = 0.0;
	}
}

static void heft_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	int workerid;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		_starpu_destroy_fifo(hd->queue_array[workerid]);
		starpu_worker_deinit_sched_condition(sched_ctx_id, workerid);
		current_time[workerid][sched_ctx_id] = 0.0;
	}
}

static void heft_init(unsigned sched_ctx_id)
{
	starpu_create_worker_collection_for_sched_ctx(sched_ctx_id, WORKER_LIST);

	heft_data *hd = (heft_data*)malloc(sizeof(heft_data));
	hd->alpha = _STARPU_DEFAULT_ALPHA;
	hd->beta = _STARPU_DEFAULT_BETA;
	hd->_gamma = _STARPU_DEFAULT_GAMMA;
	hd->idle_power = 0.0;
	
	starpu_set_sched_ctx_policy_data(sched_ctx_id, (void*)hd);

	hd->queue_array = (struct _starpu_fifo_taskq**)malloc(STARPU_NMAXWORKERS*sizeof(struct _starpu_fifo_taskq*));

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		hd->alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		hd->beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		hd->_gamma = atof(strval_gamma);

	const char *strval_idle_power = getenv("STARPU_IDLE_POWER");
	if (strval_idle_power)
		hd->idle_power = atof(strval_idle_power);

	starpu_top_register_parameter_float("HEFT_ALPHA", &hd->alpha, alpha_minimum,alpha_maximum,param_modified);
	starpu_top_register_parameter_float("HEFT_BETA", &hd->beta, beta_minimum,beta_maximum,param_modified);
	starpu_top_register_parameter_float("HEFT_GAMMA", &hd->_gamma, gamma_minimum,gamma_maximum,param_modified);
	starpu_top_register_parameter_float("HEFT_IDLE_POWER", &hd->idle_power, idle_power_minimum,idle_power_maximum,param_modified);
}


/* heft_pre_exec_hook is called right after the data transfer is done and right before
 * the computation to begin, it is useful to update more precisely the value
 * of the expected start, end, length, etc... */
static void heft_pre_exec_hook(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	int workerid = starpu_worker_get_id();
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = hd->queue_array[workerid];
	double model = task->predicted;
	double transfer_model = task->predicted_transfer;

	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, workerid, &sched_mutex, &sched_cond);
	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	fifo->exp_len-= transfer_model;
	fifo->exp_start = starpu_timing_now() + model;
	fifo->exp_end= fifo->exp_start + fifo->exp_len;
	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static void heft_push_task_notify(struct starpu_task *task, int workerid)
{
	unsigned sched_ctx_id = task->sched_ctx;
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = hd->queue_array[workerid];
	/* Compute the expected penality */
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	unsigned memory_node = starpu_worker_get_memory_node(workerid);

	double predicted = starpu_task_expected_length(task, perf_arch,
			_starpu_get_job_associated_to_task(task)->nimpl);

	double predicted_transfer = starpu_task_expected_data_transfer_time(memory_node, task);
	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, workerid, &sched_mutex, &sched_cond);


	/* Update the predictions */
	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	/* Sometimes workers didn't take the tasks as early as we expected */
	fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	/* If there is no prediction available, we consider the task has a null length */
	if (!isnan(predicted))
	{
		task->predicted = predicted;
		fifo->exp_end += predicted;
		fifo->exp_len += predicted;
	}

	/* If there is no prediction available, we consider the task has a null length */
	if (!isnan(predicted_transfer))
	{
		if (starpu_timing_now() + predicted_transfer < fifo->exp_end)
		{
			/* We may hope that the transfer will be finished by
			 * the start of the task. */
			predicted_transfer = 0;
		}
		else
		{
			/* The transfer will not be finished by then, take the
			 * remainder into account */
			predicted_transfer = (starpu_timing_now() + predicted_transfer) - fifo->exp_end;
		}
		task->predicted_transfer = predicted_transfer;
		fifo->exp_end += predicted_transfer;
		fifo->exp_len += predicted_transfer;
	}

	fifo->ntasks++;

	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, double predicted_transfer, unsigned sched_ctx_id)
 {
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = hd->queue_array[best_workerid];

	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, best_workerid, &sched_mutex, &sched_cond);

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	starpu_call_pushed_task_cb(best_workerid, sched_ctx_id);
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR

	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);

	/* Sometimes workers didn't take the tasks as early as we expected */
	fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

	fifo->exp_end += predicted;
	fifo->exp_len += predicted;

	if (starpu_timing_now() + predicted_transfer < fifo->exp_end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer = (starpu_timing_now() + predicted_transfer) - fifo->exp_end;
	}
	fifo->exp_end += predicted_transfer;
	fifo->exp_len += predicted_transfer;

	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

	task->predicted = predicted;
	task->predicted_transfer = predicted_transfer;

	if (_starpu_top_status_get())
		_starpu_top_task_prevision(task, best_workerid,
					(unsigned long long)(fifo->exp_end-predicted)/1000,
					(unsigned long long)fifo->exp_end/1000);

	if (starpu_get_prefetch_flag())
	{
		unsigned memory_node = starpu_worker_get_memory_node(best_workerid);
		starpu_prefetch_task_input_on_node(task, memory_node);
	}


	double max_time_on_ctx = starpu_get_max_time_worker_on_ctx();
	if(max_time_on_ctx != -1.0 && starpu_are_overlapping_ctxs_on_worker(best_workerid) && starpu_is_ctxs_turn(best_workerid, sched_ctx_id))
	{
		current_time[best_workerid][sched_ctx_id] += predicted;
		
		if(current_time[best_workerid][sched_ctx_id] >= max_time_on_ctx)
		{
			current_time[best_workerid][sched_ctx_id] = 0.0;
			starpu_set_turn_to_other_ctx(best_workerid, sched_ctx_id);
		}
	}

	#ifdef HAVE_AYUDAME_H
	if (AYU_event) {
		int id = best_workerid;
		AYU_event(AYU_ADDTASKTOQUEUE, _starpu_get_job_associated_to_task(task)->job_id, &id);
	}
#endif
	return _starpu_fifo_push_task(hd->queue_array[best_workerid],
				      sched_mutex, sched_cond, task);
}

/* TODO: Correct the bugs in the scheduling !!! */
/* TODO: factorize with dmda!! */
static void compute_all_performance_predictions(struct starpu_task *task,
						double local_task_length[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS], 
						double exp_end[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS],
						double *max_exp_endp, double *best_exp_endp,
						double local_data_penalty[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS],
						double local_power[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS], 
						int *forced_worker, int *forced_impl, unsigned sched_ctx_id)
{
	int calibrating = 0;
	double max_exp_end = DBL_MIN;
	double best_exp_end = DBL_MAX;
	int ntasks_best = -1;
	int nimpl_best = 0;
	double ntasks_best_end = 0.0;

	/* A priori, we know all estimations */
	int unknown = 0;
	int worker, worker_ctx = 0;
	unsigned nimpl;

	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	starpu_task_bundle_t bundle = task->bundle;
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);

	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		if(starpu_is_ctxs_turn(worker, sched_ctx_id) || sched_ctx_id == 0)
		{
			for (nimpl = 0; nimpl <STARPU_MAXIMPLEMENTATIONS; nimpl++) 
			{
				if (!starpu_worker_can_execute_task(worker, task, nimpl))
				{
					/* no one on that queue may execute this task */
//				worker_ctx++;
					continue;
				}
		
				/* Sometimes workers didn't take the tasks as early as we expected */
				struct _starpu_fifo_taskq *fifo = hd->queue_array[worker];
				pthread_mutex_t *sched_mutex;
				pthread_cond_t *sched_cond;
				starpu_worker_get_sched_condition(sched_ctx_id, worker, &sched_mutex, &sched_cond);
				_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
				fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
				exp_end[worker_ctx][nimpl] = fifo->exp_start + fifo->exp_len;
				if (exp_end[worker_ctx][nimpl] > max_exp_end)
					max_exp_end = exp_end[worker_ctx][nimpl];
				_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
				
				enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
				unsigned memory_node = starpu_worker_get_memory_node(worker);
				
				if (bundle)
				{
					/* TODO : conversion time */
					local_task_length[worker_ctx][nimpl] = starpu_task_bundle_expected_length(bundle, perf_arch, nimpl);
					local_data_penalty[worker_ctx][nimpl] = starpu_task_bundle_expected_data_transfer_time(bundle, memory_node);
					local_power[worker_ctx][nimpl] = starpu_task_bundle_expected_power(bundle, perf_arch, nimpl);
					//_STARPU_DEBUG("Scheduler heft bundle: task length (%lf) local power (%lf) worker (%u) kernel (%u) \n", local_task_length[worker_ctx],local_power[worker_ctx],worker,nimpl);
				}
				else 
				{
					local_task_length[worker_ctx][nimpl] = starpu_task_expected_length(task, perf_arch, nimpl);
					local_data_penalty[worker_ctx][nimpl] = starpu_task_expected_data_transfer_time(memory_node, task);
					local_power[worker_ctx][nimpl] = starpu_task_expected_power(task, perf_arch, nimpl);
					double conversion_time = starpu_task_expected_conversion_time(task, perf_arch, nimpl);
					if (conversion_time > 0.0)
						local_task_length[worker_ctx][nimpl] += conversion_time;

					//_STARPU_DEBUG("Scheduler heft bundle: task length (%lf) local power (%lf) worker (%u) kernel (%u) \n", local_task_length[worker_ctx],local_power[worker_ctx],worker,nimpl);
				}
				double ntasks_end = fifo->ntasks / starpu_worker_get_relative_speedup(perf_arch);

/* 				printf("**********%d/%d: len = %lf penalty = %lf \n", worker, worker_ctx,  */
/* 				       local_task_length[worker_ctx][nimpl], local_data_penalty[worker_ctx][nimpl]); */
				
				if (ntasks_best == -1
				    || (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better worker */
				    || (!calibrating && isnan(local_task_length[worker_ctx][nimpl])) /* Not calibrating but this worker is being calibrated */
				    || (calibrating && isnan(local_task_length[worker_ctx][nimpl]) && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
					)
				{
					ntasks_best_end = ntasks_end;
					ntasks_best = worker;
					nimpl_best = nimpl;
				}
				
				if (isnan(local_task_length[worker_ctx][nimpl]))
					/* we are calibrating, we want to speed-up calibration time
					 * so we privilege non-calibrated tasks (but still
					 * greedily distribute them to avoid dumb schedules) */
					calibrating = 1;
				
				if (isnan(local_task_length[worker_ctx][nimpl])
				    || _STARPU_IS_ZERO(local_task_length[worker_ctx][nimpl]))
					/* there is no prediction available for that task
					 * with that arch (yet or at all), so switch to a greedy strategy */
					unknown = 1;
				
				if (unknown)
					continue;
				
				exp_end[worker_ctx][nimpl] = fifo->exp_start + fifo->exp_len + local_task_length[worker_ctx][nimpl];
			
				if (exp_end[worker_ctx][nimpl] < best_exp_end)
				{
					/* a better solution was found */
					best_exp_end = exp_end[worker_ctx][nimpl];
					nimpl_best = nimpl;
				}
				
				if (isnan(local_power[worker_ctx][nimpl]))
					local_power[worker_ctx][nimpl] = 0.;
				
			}
		}
		worker_ctx++;
	}

	*forced_worker = unknown?ntasks_best:-1;
	*forced_impl = unknown?nimpl_best:-1;

	*best_exp_endp = best_exp_end;
	*max_exp_endp = max_exp_end;
}

/* TODO: factorize with dmda */
static int _heft_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	int worker, worker_ctx = 0;
	unsigned nimpl;
	int best = -1, best_in_ctx = -1;
	int selected_impl= -1;

	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_worker;
	int forced_impl;
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);

	unsigned nworkers_ctx = workers->nworkers;
	double local_task_length[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double local_data_penalty[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double local_power[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double exp_end[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double max_exp_end = 0.0;

	double best_exp_end;

	/*
	 *	Compute the expected end of the task on the various workers,
	 *	and detect if there is some calibration that needs to be done.
	 */

	starpu_task_bundle_t bundle = task->bundle;

	if(workers->init_cursor)
		workers->init_cursor(workers);

	compute_all_performance_predictions(task, local_task_length, exp_end,
					&max_exp_end, &best_exp_end,
					local_data_penalty,
					local_power, &forced_worker, &forced_impl,
					sched_ctx_id);

	/* If there is no prediction available for that task with that arch we
	 * want to speed-up calibration time so we force this measurement */
	if (forced_worker != -1)
	{
		_starpu_get_job_associated_to_task(task)->nimpl = forced_impl;

		if (task->bundle)
			starpu_task_bundle_remove(task->bundle, task);

		return push_task_on_best_worker(task, forced_worker, 0.0, 0.0, sched_ctx_id);
	}

	/*
	 *	Determine which worker optimizes the fitness metric which is a
	 *	trade-off between load-balacing, data locality, and energy
	 *	consumption.
	 */
	
	double fitness[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];
	double best_fitness = -1;

	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		if(starpu_is_ctxs_turn(worker, sched_ctx_id) || sched_ctx_id == 0)
		{
			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!starpu_worker_can_execute_task(worker, task, nimpl))
				{
					/* no one on that queue may execute this task */
					//worker_ctx++;
					continue;
				}
				
				
				fitness[worker_ctx][nimpl] = hd->alpha*(exp_end[worker_ctx][nimpl] - best_exp_end) 
					+ hd->beta*(local_data_penalty[worker_ctx][nimpl])
					+ hd->_gamma*(local_power[worker_ctx][nimpl]);
				
				if (exp_end[worker_ctx][nimpl] > max_exp_end)
					/* This placement will make the computation
					 * longer, take into account the idle
				 * consumption of other cpus */
					fitness[worker_ctx][nimpl] += hd->_gamma * hd->idle_power * (exp_end[worker_ctx][nimpl] - max_exp_end) / 1000000.0;
			
				if (best == -1 || fitness[worker_ctx][nimpl] < best_fitness)
				{
					/* we found a better solution */
					best_fitness = fitness[worker_ctx][nimpl];
					best = worker;
					best_in_ctx = worker_ctx;
					selected_impl = nimpl;
				}
			}
		}
		worker_ctx++;
	}

	if(best == -1)
		return -1;

	/* By now, we must have found a solution */
	STARPU_ASSERT(best != -1);

	/* we should now have the best worker in variable "best" */
	double model_best, transfer_model_best;

	if (bundle)
	{
		/* If we have a task bundle, we have computed the expected
		 * length for the entire bundle, but not for the task alone. */
		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(best);
		unsigned memory_node = starpu_worker_get_memory_node(best);
		model_best = starpu_task_expected_length(task, perf_arch, selected_impl);
		transfer_model_best = starpu_task_expected_data_transfer_time(memory_node, task);

		/* Remove the task from the bundle since we have made a
		 * decision for it, and that other tasks should not consider it
		 * anymore. */
		starpu_task_bundle_remove(bundle, task);
	}
	else 
	{
		model_best = local_task_length[best_in_ctx][selected_impl];
		transfer_model_best = local_data_penalty[best_in_ctx][selected_impl];
	}

	if(workers->init_cursor)
		workers->deinit_cursor(workers);

	_starpu_get_job_associated_to_task(task)->nimpl = selected_impl;

	return push_task_on_best_worker(task, best, model_best, transfer_model_best, sched_ctx_id);
}

static int heft_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	pthread_mutex_t *changing_ctx_mutex = starpu_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers; 
	int ret_val = -1;
	if (task->priority > 0)
	{
		_STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
		nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
		if(nworkers == 0)
		{
			_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
			return ret_val;
		}

		ret_val = _heft_push_task(task, 1, sched_ctx_id);
		_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	_STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
	nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
	if(nworkers == 0)
	{
		_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	ret_val = _heft_push_task(task, 0, sched_ctx_id);
	_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);

	return ret_val;
}

static struct starpu_task *heft_pop_task(unsigned sched_ctx_id)
{
	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = hd->queue_array[workerid];

	task = _starpu_fifo_pop_local_task(fifo);
	if (task)
	{
		double model = task->predicted;

		fifo->exp_len -= model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}

	return task;
}

static void heft_deinit(unsigned sched_ctx_id) 
{
	heft_data *ht = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	free(ht);
	starpu_delete_worker_collection_for_sched_ctx(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_heft_policy =
{
	.init_sched = heft_init,
	.deinit_sched = heft_deinit,
	.push_task = heft_push_task,
	.push_task_notify = heft_push_task_notify,
	.pop_task = NULL,
	.pop_every_task = NULL,
	.pre_exec_hook = heft_pre_exec_hook,
	.post_exec_hook = NULL,
	.add_workers = heft_add_workers	,
	.remove_workers = heft_remove_workers,
	.policy_name = "heft",
	.policy_description = "Heterogeneous Earliest Finish Task"
};
