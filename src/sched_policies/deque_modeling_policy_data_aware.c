/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2011-2012  INRIA
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
#include <limits.h>

#include <core/perfmodel/perfmodel.h>
#include <core/task_bundle.h>
#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <core/perfmodel/perfmodel.h>
#include <starpu_parameters.h>
#include <core/debug.h>
#ifdef STARPU_USE_TOP
#include <top/starpu_top_core.h>
#endif /* !STARPU_USE_TOP */

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

	struct _starpu_fifo_taskq **queue_array;

	long int total_task_cnt;
	long int ready_task_cnt;
};

static double alpha = _STARPU_DEFAULT_ALPHA;
static double beta = _STARPU_DEFAULT_BETA;
static double _gamma = _STARPU_DEFAULT_GAMMA;
static double idle_power = 0.0;

#ifdef STARPU_USE_TOP
static const float alpha_minimum=0;
static const float alpha_maximum=10.0;
static const float beta_minimum=0;
static const float beta_maximum=10.0;
static const float gamma_minimum=0;
static const float gamma_maximum=10000.0;
static const float idle_power_minimum=0;
static const float idle_power_maximum=10000.0;
#endif /* !STARPU_USE_TOP */

static int count_non_ready_buffers(struct starpu_task *task, unsigned node)
{
	int cnt = 0;
	unsigned nbuffers = task->cl->nbuffers;
	unsigned index;

	for (index = 0; index < nbuffers; index++)
	{
		starpu_data_handle_t handle;

		handle = task->handles[index];

		int is_valid;
		starpu_data_query_status(handle, node, NULL, &is_valid, NULL);

		if (!is_valid)
			cnt++;
	}

	return cnt;
}

#ifdef STARPU_USE_TOP
static void param_modified(struct starpu_top_param* d)
{
#ifdef STARPU_DEVEL
#warning FIXME: get sched ctx to get alpha/beta/gamma/idle values
#endif
	/* Just to show parameter modification. */
	fprintf(stderr,
		"%s has been modified : "
		"alpha=%f|beta=%f|gamma=%f|idle_power=%f !\n",
		d->name, alpha,beta,_gamma, idle_power);
}
#endif /* !STARPU_USE_TOP */

static struct starpu_task *_starpu_fifo_pop_first_ready_task(struct _starpu_fifo_taskq *fifo_queue, unsigned node)
{
	struct starpu_task *task = NULL, *current;

	if (fifo_queue->ntasks == 0)
		return NULL;

	if (fifo_queue->ntasks > 0)
	{
		fifo_queue->ntasks--;

		task = starpu_task_list_back(&fifo_queue->taskq);
		if (STARPU_UNLIKELY(!task))
			return NULL;

		int first_task_priority = task->priority;

		current = task;

		int non_ready_best = INT_MAX;

		while (current)
		{
			int priority = current->priority;

			if (priority <= first_task_priority)
			{
				int non_ready = count_non_ready_buffers(current, node);
				if (non_ready < non_ready_best)
				{
					non_ready_best = non_ready;
					task = current;

					if (non_ready == 0)
						break;
				}
			}

			current = current->prev;
		}

		starpu_task_list_erase(&fifo_queue->taskq, task);

		_STARPU_TRACE_JOB_POP(task, 0);
	}

	return task;
}

static struct starpu_task *dmda_pop_ready_task(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	struct _starpu_fifo_taskq *fifo = dt->queue_array[workerid];

	unsigned node = starpu_worker_get_memory_node(workerid);

	task = _starpu_fifo_pop_first_ready_task(fifo, node);
	if (task)
	{
		double model = task->predicted;

		if(!isnan(model))
		{
			fifo->exp_len -= model;
			fifo->exp_start = starpu_timing_now() + model;
			fifo->exp_end = fifo->exp_start + fifo->exp_len;
		}

#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = count_non_ready_buffers(task, node);
			if (non_ready == 0)
				dt->ready_task_cnt++;
		}

		dt->total_task_cnt++;
#endif
	}

	return task;
}

static struct starpu_task *dmda_pop_task(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task;

	int workerid = starpu_worker_get_id();
	struct _starpu_fifo_taskq *fifo = dt->queue_array[workerid];

	STARPU_ASSERT_MSG(fifo, "worker %d does not belong to ctx %d anymore \n", workerid, sched_ctx_id);

	task = _starpu_fifo_pop_local_task(fifo);
	if (task)
	{
		double model = task->predicted;
		if(!isnan(model))
		{
			fifo->exp_len -= model;
			fifo->exp_start = starpu_timing_now() + model;
			fifo->exp_end = fifo->exp_start + fifo->exp_len;
		}
		  
#ifdef STARPU_VERBOSE
		if (task->cl)
		{
			int non_ready = count_non_ready_buffers(task, starpu_worker_get_memory_node(workerid));
			if (non_ready == 0)
				dt->ready_task_cnt++;
		}
		
		dt->total_task_cnt++;
#endif
	}

	return task;
}

static struct starpu_task *dmda_pop_every_task(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *new_list;

	int workerid = starpu_worker_get_id();
	struct _starpu_fifo_taskq *fifo = dt->queue_array[workerid];

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	new_list = _starpu_fifo_pop_every_task(fifo, workerid);
	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
	while (new_list)
	{
		double model = new_list->predicted;
		if(!isnan(model)) 
		{
			fifo->exp_len -= model;
			fifo->exp_start = starpu_timing_now() + model;
			fifo->exp_end = fifo->exp_start + fifo->exp_len;
		}

		new_list = new_list->next;
	}

	return new_list;
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid,
				    double predicted, double predicted_transfer,
				    int prio, unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	struct _starpu_fifo_taskq *fifo = dt->queue_array[best_workerid];

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(best_workerid, &sched_mutex, &sched_cond);

#ifdef STARPU_USE_SC_HYPERVISOR
	starpu_sched_ctx_call_pushed_task_cb(best_workerid, sched_ctx_id);
#endif //STARPU_USE_SC_HYPERVISOR

	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);

/* Sometimes workers didn't take the tasks as early as we expected */
	fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
	fifo->exp_end = fifo->exp_start + fifo->exp_len;

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
		predicted_transfer += starpu_timing_now();
		predicted_transfer -= fifo->exp_end;
	}
	if(!isnan(predicted_transfer)) 
	{
		fifo->exp_end += predicted_transfer;
		fifo->exp_len += predicted_transfer;
	}

	if(!isnan(predicted))
	{
		fifo->exp_end += predicted;
		fifo->exp_len += predicted;
	}

	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

	task->predicted = predicted;
	task->predicted_transfer = predicted_transfer;

#ifdef STARPU_USE_TOP
	if (_starpu_top_status_get())
		_starpu_top_task_prevision(task, best_workerid,
			(unsigned long long)(fifo->exp_end-predicted)/1000,
			(unsigned long long)fifo->exp_end/1000);
#endif /* !STARPU_USE_TOP */

	if (starpu_get_prefetch_flag())
	{
		unsigned memory_node = starpu_worker_get_memory_node(best_workerid);
		starpu_prefetch_task_input_on_node(task, memory_node);
	}

#ifdef HAVE_AYUDAME_H
	if (AYU_event)
	{
		int id = best_workerid;
		AYU_event(AYU_ADDTASKTOQUEUE, _starpu_get_job_associated_to_task(task)->job_id, &id);
	}
#endif
	int ret = 0;
	if (prio)
	{
		_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		ret =_starpu_fifo_push_sorted_task(dt->queue_array[best_workerid], task);
		_STARPU_PTHREAD_COND_SIGNAL(sched_cond);
		starpu_push_task_end(task);
		_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
	}
	else
	{
		_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		ret = _starpu_fifo_push_task(dt->queue_array[best_workerid], task);
		_STARPU_PTHREAD_COND_SIGNAL(sched_cond);
		starpu_push_task_end(task);
		_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
	}

	return ret;
}

/* TODO: factorize with dmda!! */
static int _dm_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned worker, worker_ctx = 0;
	int best = -1;

	double best_exp_end = 0.0;
	double model_best = 0.0;
	double transfer_model_best = 0.0;

	int ntasks_best = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;

	/* A priori, we know all estimations */
	int unknown = 0;

	unsigned best_impl = 0;
	unsigned nimpl;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		struct _starpu_fifo_taskq *fifo  = dt->queue_array[worker];
		unsigned memory_node = starpu_worker_get_memory_node(worker);
		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);

		/* Sometimes workers didn't take the tasks as early as we expected */
		_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
		_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);


		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!starpu_worker_can_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				//			worker_ctx++;
				continue;
			}

			double exp_end;
			starpu_pthread_mutex_t *sched_mutex;
			starpu_pthread_cond_t *sched_cond;
			starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);

			double local_length = starpu_task_expected_length(task, perf_arch, nimpl);
			double local_penalty = starpu_task_expected_data_transfer_time(memory_node, task);
			double ntasks_end = fifo->ntasks / starpu_worker_get_relative_speedup(perf_arch);

			//_STARPU_DEBUG("Scheduler dm: task length (%lf) worker (%u) kernel (%u) \n", local_length,worker,nimpl);

			if (ntasks_best == -1
			    || (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
			    || (!calibrating && isnan(local_length)) /* Not calibrating but this worker is being calibrated */
			    || (calibrating && isnan(local_length) && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
				)
			{
				ntasks_best_end = ntasks_end;
				ntasks_best = worker;
				best_impl = nimpl;
			}

			if (isnan(local_length))
				/* we are calibrating, we want to speed-up calibration time
				 * so we privilege non-calibrated tasks (but still
				 * greedily distribute them to avoid dumb schedules) */
				calibrating = 1;

			if (isnan(local_length) || _STARPU_IS_ZERO(local_length))
				/* there is no prediction available for that task
				 * with that arch yet, so switch to a greedy strategy */
				unknown = 1;

			if (unknown)
				continue;

			exp_end = fifo->exp_start + fifo->exp_len + local_length;

			if (best == -1 || exp_end < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = exp_end;
				best = worker;
				model_best = local_length;
				transfer_model_best = local_penalty;
				best_impl = nimpl;
			}
		}
		worker_ctx++;
	}

	if (unknown)
	{
		best = ntasks_best;
		model_best = 0.0;
		transfer_model_best = 0.0;
	}

	//_STARPU_DEBUG("Scheduler dm: kernel (%u)\n", best_impl);

	_starpu_get_job_associated_to_task(task)->nimpl = best_impl;

	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best,
					model_best, transfer_model_best, prio, sched_ctx_id);
}

/* TODO: factorise CPU computations, expensive with a lot of cores */
static void compute_all_performance_predictions(struct starpu_task *task,
						double local_task_length[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS],
						double exp_end[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS],
						double *max_exp_endp,
						double *best_exp_endp,
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
	unsigned worker, worker_ctx = 0;

	unsigned nimpl;

	starpu_task_bundle_t bundle = task->bundle;
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);

	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		struct _starpu_fifo_taskq *fifo = dt->queue_array[worker];
		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
		unsigned memory_node = starpu_worker_get_memory_node(worker);

		_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
		_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

		for(nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
	 	{
			if (!starpu_worker_can_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				continue;
			}

			/* Sometimes workers didn't take the tasks as early as we expected */
			starpu_pthread_mutex_t *sched_mutex;
			starpu_pthread_cond_t *sched_cond;
			starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);

			STARPU_ASSERT_MSG(fifo != NULL, "worker %d ctx %d\n", worker, sched_ctx_id);
			exp_end[worker_ctx][nimpl] = fifo->exp_start + fifo->exp_len;
			if (exp_end[worker_ctx][nimpl] > max_exp_end)
				max_exp_end = exp_end[worker_ctx][nimpl];

			//_STARPU_DEBUG("Scheduler dmda: task length (%lf) worker (%u) kernel (%u) \n", local_task_length[worker][nimpl],worker,nimpl);

			if (bundle)
			{
				/* TODO : conversion time */
				local_task_length[worker_ctx][nimpl] = starpu_task_bundle_expected_length(bundle, perf_arch, nimpl);
				local_data_penalty[worker_ctx][nimpl] = starpu_task_bundle_expected_data_transfer_time(bundle, memory_node);
				local_power[worker_ctx][nimpl] = starpu_task_bundle_expected_power(bundle, perf_arch,nimpl);
			}
			else
			{
				local_task_length[worker_ctx][nimpl] = starpu_task_expected_length(task, perf_arch, nimpl);
				local_data_penalty[worker_ctx][nimpl] = starpu_task_expected_data_transfer_time(memory_node, task);
				local_power[worker_ctx][nimpl] = starpu_task_expected_power(task, perf_arch,nimpl);
				double conversion_time = starpu_task_expected_conversion_time(task, perf_arch, nimpl);
				if (conversion_time > 0.0)
					local_task_length[worker_ctx][nimpl] += conversion_time;
			}
			
			double ntasks_end = fifo->ntasks / starpu_worker_get_relative_speedup(perf_arch);

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
		worker_ctx++;
	}

	*forced_worker = unknown?ntasks_best:-1;
	*forced_impl = unknown?nimpl_best:-1;

	*best_exp_endp = best_exp_end;
	*max_exp_endp = max_exp_end;
}

static int _dmda_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	/* find the queue */
	unsigned worker, worker_ctx = 0;
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
	unsigned nworkers_ctx = workers->nworkers;
	double local_task_length[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double local_data_penalty[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double local_power[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double exp_end[STARPU_NMAXWORKERS][STARPU_MAXIMPLEMENTATIONS];
	double max_exp_end = 0.0;
	double best_exp_end;

	double fitness[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];

	struct starpu_sched_ctx_iterator it;
	if(workers->init_iterator)
		workers->init_iterator(workers, &it);


	compute_all_performance_predictions(task,
					    local_task_length,
					    exp_end,
					    &max_exp_end,
					    &best_exp_end,
					    local_data_penalty,
					    local_power,
					    &forced_best,
					    &forced_impl, sched_ctx_id);
	
	double best_fitness = -1;

	unsigned nimpl;
	if (forced_best == -1)
	{
		while(workers->has_next(workers, &it))
		{
			worker = workers->get_next(workers, &it);
			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!starpu_worker_can_execute_task(worker, task, nimpl))
				{
					/* no one on that queue may execute this task */
					continue;
				}


				fitness[worker_ctx][nimpl] = dt->alpha*(exp_end[worker_ctx][nimpl] - best_exp_end)
					+ dt->beta*(local_data_penalty[worker_ctx][nimpl])
					+ dt->_gamma*(local_power[worker_ctx][nimpl]);

				if (exp_end[worker_ctx][nimpl] > max_exp_end)
				{
					/* This placement will make the computation
					 * longer, take into account the idle
					 * consumption of other cpus */
					fitness[worker_ctx][nimpl] += dt->_gamma * dt->idle_power * (exp_end[worker_ctx][nimpl] - max_exp_end) / 1000000.0;
				}

				if (best == -1 || fitness[worker_ctx][nimpl] < best_fitness)
				{
					/* we found a better solution */
					best_fitness = fitness[worker_ctx][nimpl];
					best = worker;
					best_in_ctx = worker_ctx;
					selected_impl = nimpl;

					//_STARPU_DEBUG("best fitness (worker %d) %e = alpha*(%e) + beta(%e) +gamma(%e)\n", worker, best_fitness, exp_end[worker][nimpl] - best_exp_end, local_data_penalty[worker][nimpl], local_power[worker][nimpl]);
				}
			}
			worker_ctx++;
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
		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(best_in_ctx);
		unsigned memory_node = starpu_worker_get_memory_node(best);
		model_best = starpu_task_expected_length(task, perf_arch, selected_impl);
		transfer_model_best = starpu_task_expected_data_transfer_time(memory_node, task);
	}
	else
	{
		model_best = local_task_length[best_in_ctx][selected_impl];
		transfer_model_best = local_data_penalty[best_in_ctx][selected_impl];
	}

	//_STARPU_DEBUG("Scheduler dmda: kernel (%u)\n", best_impl);
	 _starpu_get_job_associated_to_task(task)->nimpl = selected_impl;

	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, model_best, transfer_model_best, prio, sched_ctx_id);
}

static int dmda_push_sorted_task(struct starpu_task *task)
{
#ifdef STARPU_DEVEL
#warning TODO: after defining a scheduling window, use that instead of empty_ctx_tasks
#endif
	unsigned sched_ctx_id = task->sched_ctx;
	starpu_pthread_mutex_t *changing_ctx_mutex = starpu_sched_ctx_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers;
	int ret_val = -1;

	_STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
	nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
	if(nworkers == 0)
	{
		_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	ret_val = _dmda_push_task(task, 1, sched_ctx_id);
	_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
	return ret_val;

}

static int dm_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	starpu_pthread_mutex_t *changing_ctx_mutex = starpu_sched_ctx_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers;
	int ret_val = -1;

	_STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
	nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
	if(nworkers == 0)
	{
		_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	ret_val = _dm_push_task(task, 0, sched_ctx_id);
	_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
	return ret_val;
}

static int dmda_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	starpu_pthread_mutex_t *changing_ctx_mutex = starpu_sched_ctx_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers;
	int ret_val = -1;

	_STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
	nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
	if(nworkers == 0)
	{
		_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	STARPU_ASSERT(task);
	ret_val = _dmda_push_task(task, 0, sched_ctx_id);
	_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
	return ret_val;
}

static void dmda_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int workerid;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		/* if the worker has alreadry belonged to this context
		   the queue and the synchronization variables have been already initialized */
		if(dt->queue_array[workerid] == NULL)
			dt->queue_array[workerid] = _starpu_create_fifo();
	}
}

static void dmda_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int workerid;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		if(dt->queue_array[workerid] != NULL)
		{
			_starpu_destroy_fifo(dt->queue_array[workerid]);
			dt->queue_array[workerid] = NULL;
		}
	}
}

static void initialize_dmda_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);

	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)malloc(sizeof(struct _starpu_dmda_data));
	dt->alpha = _STARPU_DEFAULT_ALPHA;
	dt->beta = _STARPU_DEFAULT_BETA;
	dt->_gamma = _STARPU_DEFAULT_GAMMA;
	dt->idle_power = 0.0;

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)dt);

	dt->queue_array = (struct _starpu_fifo_taskq**)malloc(STARPU_NMAXWORKERS*sizeof(struct _starpu_fifo_taskq*));

	int i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		dt->queue_array[i] = NULL;

	const char *strval_alpha = getenv("STARPU_SCHED_ALPHA");
	if (strval_alpha)
		dt->alpha = atof(strval_alpha);

	const char *strval_beta = getenv("STARPU_SCHED_BETA");
	if (strval_beta)
		dt->beta = atof(strval_beta);

	const char *strval_gamma = getenv("STARPU_SCHED_GAMMA");
	if (strval_gamma)
		dt->_gamma = atof(strval_gamma);

	const char *strval_idle_power = getenv("STARPU_IDLE_POWER");
	if (strval_idle_power)
		dt->idle_power = atof(strval_idle_power);

#ifdef STARPU_USE_TOP
	starpu_top_register_parameter_float("DMDA_ALPHA", &alpha,
		alpha_minimum, alpha_maximum, param_modified);
	starpu_top_register_parameter_float("DMDA_BETA", &beta,
		beta_minimum, beta_maximum, param_modified);
	starpu_top_register_parameter_float("DMDA_GAMMA", &_gamma,
		gamma_minimum, gamma_maximum, param_modified);
	starpu_top_register_parameter_float("DMDA_IDLE_POWER", &idle_power,
		idle_power_minimum, idle_power_maximum, param_modified);
#endif /* !STARPU_USE_TOP */
}

static void initialize_dmda_sorted_policy(unsigned sched_ctx_id)
{
	initialize_dmda_policy(sched_ctx_id);

	/* The application may use any integer */
	starpu_sched_ctx_set_min_priority(sched_ctx_id, INT_MIN);
	starpu_sched_ctx_set_max_priority(sched_ctx_id, INT_MAX);
}

static void deinitialize_dmda_policy(unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	_STARPU_DEBUG("total_task_cnt %ld ready_task_cnt %ld -> %f\n", dt->total_task_cnt, dt->ready_task_cnt, (100.0f*dt->ready_task_cnt)/dt->total_task_cnt);

	free(dt->queue_array);
	free(dt);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

/* dmda_pre_exec_hook is called right after the data transfer is done and right
 * before the computation to begin, it is useful to update more precisely the
 * value of the expected start, end, length, etc... */
static void dmda_pre_exec_hook(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	int workerid = starpu_worker_get_id();
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = dt->queue_array[workerid];
	double model = task->predicted;
	double transfer_model = task->predicted_transfer;

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);

	/* Once the task is executing, we can update the predicted amount
	 * of work. */
	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	if(!isnan(transfer_model))
	{
		fifo->exp_len-= transfer_model;
		fifo->exp_start = starpu_timing_now() + model;
		fifo->exp_end= fifo->exp_start + fifo->exp_len;
	}
	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static void dmda_push_task_notify(struct starpu_task *task, int workerid, unsigned sched_ctx_id)
{
	struct _starpu_dmda_data *dt = (struct _starpu_dmda_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = dt->queue_array[workerid];
	/* Compute the expected penality */
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	unsigned memory_node = starpu_worker_get_memory_node(workerid);

	double predicted = starpu_task_expected_length(task, perf_arch,
			_starpu_get_job_associated_to_task(task)->nimpl);

	double predicted_transfer = starpu_task_expected_data_transfer_time(memory_node, task);
	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);


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

/* TODO: use post_exec_hook to fix the expected start */
struct starpu_sched_policy _starpu_sched_dm_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dm_push_task,
	.pop_task = dmda_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dm",
	.policy_description = "performance model"
};

struct starpu_sched_policy _starpu_sched_dmda_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmda",
	.policy_description = "data-aware performance model"
};

struct starpu_sched_policy _starpu_sched_dmda_sorted_policy =
{
	.init_sched = initialize_dmda_sorted_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_sorted_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_ready_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmdas",
	.policy_description = "data-aware performance model (sorted)"
};

struct starpu_sched_policy _starpu_sched_dmda_ready_policy =
{
	.init_sched = initialize_dmda_policy,
	.deinit_sched = deinitialize_dmda_policy,
	.add_workers = dmda_add_workers ,
	.remove_workers = dmda_remove_workers,
	.push_task = dmda_push_task,
	.push_task_notify = dmda_push_task_notify,
	.pop_task = dmda_pop_ready_task,
	.pre_exec_hook = dmda_pre_exec_hook,
	.post_exec_hook = NULL,
	.pop_every_task = dmda_pop_every_task,
	.policy_name = "dmdar",
	.policy_description = "data-aware performance model (ready)"
};
