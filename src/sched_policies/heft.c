/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include <starpu_parameters.h>
#include <starpu_task_bundle.h>
#include <starpu_top.h>

typedef struct {
	double alpha;
	double beta;
	double _gamma;
	double idle_power;
} heft_data;

static double exp_start[STARPU_NMAXWORKERS];	/* of the first queued task */
static double exp_end[STARPU_NMAXWORKERS];	/* of the set of queued tasks */
static double exp_len[STARPU_NMAXWORKERS];	/* of the last queued task */
static double ntasks[STARPU_NMAXWORKERS];


const float alpha_minimum=0;
const float alpha_maximum=10.0;
const float beta_minimum=0;
const float beta_maximum=10.0;
const float gamma_minimum=0;
const float gamma_maximum=10000.0;
const float idle_power_minimum=0;
const float idle_power_maximum=10000.0;

void param_modified(struct starputop_param_t* d){
	//just to show parameter modification
	fprintf(stderr,"%s has been modified : %f !\n", d->name, d->value);
}


static void heft_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nnew_workers)
{
	int workerid;
	unsigned i;
	for (i = 0; i < nnew_workers; i++)
	{
		workerid = workerids[i];
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		/* init these structures only once for each worker */
		if(!workerarg->has_prev_init)
		{
			exp_start[workerid] = starpu_timing_now();
			exp_len[workerid] = 0.0;
			exp_end[workerid] = exp_start[workerid]; 
			ntasks[workerid] = 0;
			workerarg->has_prev_init = 1;
		}
		
		/* we push the tasks on the local lists of the workers
		   therefore the synchronisations mechanisms of the strategy
		   are the global ones */
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, &workerarg->sched_mutex, &workerarg->sched_cond);
	}
}

static void heft_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	int workerid;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, NULL, NULL);
	}
}

static void heft_init(unsigned sched_ctx_id)
{
	starpu_create_worker_collection_for_sched_ctx(sched_ctx_id, WORKER_LIST);

	heft_data *hd = (heft_data*)malloc(sizeof(heft_data));
	hd->alpha = STARPU_DEFAULT_ALPHA;
	hd->beta = STARPU_DEFAULT_BETA;
	hd->_gamma = STARPU_DEFAULT_GAMMA;
	hd->idle_power = 0.0;
	
	starpu_set_sched_ctx_policy_data(sched_ctx_id, (void*)hd);

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

	starputop_register_parameter_float("HEFT_ALPHA", &hd->alpha, alpha_minimum,alpha_maximum,param_modified);
	starputop_register_parameter_float("HEFT_BETA", &hd->beta, beta_minimum,beta_maximum,param_modified);
	starputop_register_parameter_float("HEFT_GAMMA", &hd->_gamma, gamma_minimum,gamma_maximum,param_modified);
	starputop_register_parameter_float("HEFT_IDLE_POWER", &hd->idle_power, idle_power_minimum,idle_power_maximum,param_modified);
}

static void heft_post_exec_hook(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	int workerid = starpu_worker_get_id();
	STARPU_ASSERT(workerid >= 0);

	double model = task->predicted;

	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, workerid, &sched_mutex, &sched_cond);
	/* the tasks concerning changings of the the ctxs were not executed in order */
	if(!sched_mutex)
	{
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		sched_mutex = &workerarg->sched_mutex;
		sched_cond = &workerarg->sched_cond;
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, sched_mutex, sched_cond);
	}
#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	starpu_call_poped_task_cb(workerid, sched_ctx_id);
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR

	/* Once we have executed the task, we can update the predicted amount
	 * of work. */
	PTHREAD_MUTEX_LOCK(sched_mutex);
	exp_len[workerid] -= model;
	exp_start[workerid] = starpu_timing_now() + model;
	exp_end[workerid] = exp_start[workerid] + exp_len[workerid];
	ntasks[workerid]--;
	PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static void heft_push_task_notify(struct starpu_task *task, int workerid)
{
	unsigned sched_ctx_id = task->sched_ctx;
	/* Compute the expected penality */
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);

	double predicted = starpu_task_expected_length(task, perf_arch,
			_starpu_get_job_associated_to_task(task)->nimpl);

	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, workerid, &sched_mutex, &sched_cond);
	/* the tasks concerning changings of the the ctxs were not executed in order */
	if(!sched_mutex)
	{
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		sched_mutex = &workerarg->sched_mutex;
		sched_cond = &workerarg->sched_cond;
		starpu_worker_set_sched_condition(sched_ctx_id, workerid, sched_mutex, sched_cond);
	}

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	starpu_call_pushed_task_cb(workerid, sched_ctx_id);
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR

	/* Update the predictions */
	PTHREAD_MUTEX_LOCK(sched_mutex);
	/* Sometimes workers didn't take the tasks as early as we expected */
	exp_start[workerid] = STARPU_MAX(exp_start[workerid], starpu_timing_now());
	exp_end[workerid] = STARPU_MAX(exp_start[workerid], starpu_timing_now());

	/* If there is no prediction available, we consider the task has a null length */
	if (predicted != -1.0)
	{
		task->predicted = predicted;
		exp_end[workerid] += predicted;
		exp_len[workerid] += predicted;
	}

	ntasks[workerid]++;

	PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double predicted, int prio, unsigned sched_ctx_id)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	_starpu_increment_nsubmitted_tasks_of_worker(best_workerid);

	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, best_workerid, &sched_mutex, &sched_cond);
	/* the tasks concerning changings of the the ctxs were not executed in order */
	if(!sched_mutex)
	{
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(best_workerid);
		sched_mutex = &workerarg->sched_mutex;
		sched_cond = &workerarg->sched_cond;
		starpu_worker_set_sched_condition(sched_ctx_id, best_workerid, sched_mutex, sched_cond);
	}

#ifdef STARPU_USE_SCHED_CTX_HYPERVISOR
	starpu_call_pushed_task_cb(best_workerid, sched_ctx_id);
#endif //STARPU_USE_SCHED_CTX_HYPERVISOR

	PTHREAD_MUTEX_LOCK(sched_mutex);
	exp_end[best_workerid] += predicted;
	exp_len[best_workerid] += predicted;
	ntasks[best_workerid]++;
	PTHREAD_MUTEX_UNLOCK(sched_mutex);

	task->predicted = predicted;

	if (starpu_top_status_get())
		starputop_task_prevision(task, best_workerid, 
					(unsigned long long)(exp_end[best_workerid]-predicted)/1000,
					(unsigned long long)exp_end[best_workerid]/1000);

	if (starpu_get_prefetch_flag())
	{
		unsigned memory_node = starpu_worker_get_memory_node(best_workerid);
		starpu_prefetch_task_input_on_node(task, memory_node);
	}

	return starpu_push_local_task(best_workerid, task, prio);
}

static void compute_all_performance_predictions(struct starpu_task *task,
					double *local_task_length, double *exp_end,
					double *max_exp_endp, double *best_exp_endp,
					double *local_data_penalty,
					double *local_power, int *forced_best,
					struct starpu_task_bundle *bundle,
					unsigned sched_ctx_id)
{
	int calibrating = 0;
	double max_exp_end = DBL_MIN;
	double best_exp_end = DBL_MAX;
	int ntasks_best = -1;
	double ntasks_best_end = 0.0;
	
	/* A priori, we know all estimations */
	int unknown = 0;
	
	unsigned nimpl;
	unsigned best_impl = 0;
	unsigned worker, worker_ctx = 0;

	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);

	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);
		for (nimpl = 0; nimpl <STARPU_MAXIMPLEMENTATIONS; nimpl++) 
		{
			/* Sometimes workers didn't take the tasks as early as we expected */
			exp_start[worker] = STARPU_MAX(exp_start[worker], starpu_timing_now());
			exp_end[worker_ctx] = exp_start[worker] + exp_len[worker];
			if (exp_end[worker_ctx] > max_exp_end)
 				max_exp_end = exp_end[worker_ctx];
			
			if (!starpu_worker_may_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				continue;
			}
			
			enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);
			unsigned memory_node = starpu_worker_get_memory_node(worker);
			
			if (bundle)
			{
				local_task_length[worker_ctx] = starpu_task_bundle_expected_length(bundle, perf_arch, nimpl);
				local_data_penalty[worker_ctx] = starpu_task_bundle_expected_data_transfer_time(bundle, memory_node);
				local_power[worker_ctx] = starpu_task_bundle_expected_power(bundle, perf_arch, nimpl);
				//_STARPU_DEBUG("Scheduler heft bundle: task length (%lf) local power (%lf) worker (%u) kernel (%u) \n", local_task_length[worker_ctx],local_power[worker_ctx],worker,nimpl);
			}
			else 
			{
				local_task_length[worker_ctx] = starpu_task_expected_length(task, perf_arch, nimpl);
				local_data_penalty[worker_ctx] = starpu_task_expected_data_transfer_time(memory_node, task);
				local_power[worker_ctx] = starpu_task_expected_power(task, perf_arch, nimpl);
				//_STARPU_DEBUG("Scheduler heft bundle: task length (%lf) local power (%lf) worker (%u) kernel (%u) \n", local_task_length[worker_ctx],local_power[worker_ctx],worker,nimpl);
			}
			
			double ntasks_end = ntasks[worker] / starpu_worker_get_relative_speedup(perf_arch);
			
			if (ntasks_best == -1
			    || (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
			    || (!calibrating && local_task_length[worker_ctx] == -1.0) /* Not calibrating but this worker is being calibrated */
			    || (calibrating && local_task_length[worker_ctx] == -1.0 && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
				) 
			{
				ntasks_best_end = ntasks_end;
				ntasks_best = worker;
			}
			
			if (local_task_length[worker_ctx] == -1.0)
				/* we are calibrating, we want to speed-up calibration time
				 * so we privilege non-calibrated tasks (but still
				 * greedily distribute them to avoid dumb schedules) */
				calibrating = 1;
			
			if (local_task_length[worker_ctx] <= 0.0)
				/* there is no prediction available for that task
				 * with that arch yet, so switch to a greedy strategy */
				unknown = 1;
			
			if (unknown)
				continue;

			exp_end[worker_ctx] = exp_start[worker] + exp_len[worker] + local_task_length[worker_ctx];
			
			if (exp_end[worker_ctx] < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = exp_end[worker_ctx];
				best_impl = nimpl;
			}
			
			if (local_power[worker_ctx] == -1.0)
				local_power[worker_ctx] = 0.;
		}
		worker_ctx++;
	}

	*forced_best = unknown?ntasks_best:-1;

	*best_exp_endp = best_exp_end;
	*max_exp_endp = max_exp_end;
	
	/* save the best implementation */
	//_STARPU_DEBUG("Scheduler heft: kernel (%u)\n", best_impl);
	_starpu_get_job_associated_to_task(task)->nimpl = best_impl;
}

static int _heft_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	heft_data *hd = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	unsigned worker, worker_ctx = 0;
	int best = -1, best_id_ctx = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best;
	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);

	unsigned nworkers_ctx = workers->nworkers;
	double local_task_length[nworkers_ctx];
	double local_data_penalty[nworkers_ctx];
	double local_power[nworkers_ctx];
	double exp_end[nworkers_ctx];
	double max_exp_end = 0.0;

	double best_exp_end;

	/*
	 *	Compute the expected end of the task on the various workers,
	 *	and detect if there is some calibration that needs to be done.
	 */

	struct starpu_task_bundle *bundle = task->bundle;

	if(workers->init_cursor)
		workers->init_cursor(workers);

	compute_all_performance_predictions(task, local_task_length, exp_end,
					    &max_exp_end, &best_exp_end,
					    local_data_penalty,
					    local_power, &forced_best, bundle, sched_ctx_id);
	
	/* If there is no prediction available for that task with that arch we
	 * want to speed-up calibration time so we force this measurement */
	if (forced_best != -1){
		return push_task_on_best_worker(task, forced_best, 0.0, prio, sched_ctx_id);
	}
	
	/*
	 *	Determine which worker optimizes the fitness metric which is a
	 *	trade-off between load-balacing, data locality, and energy
	 *	consumption.
	 */
	
	double fitness[nworkers_ctx];
	double best_fitness = -1;

	while(workers->has_next(workers))
	{
		worker = workers->get_next(workers);

		if (!starpu_worker_may_execute_task(worker, task, 0))
		{
			/* no one on that queue may execute this task */
			continue;
		}

		fitness[worker_ctx] = hd->alpha*(exp_end[worker_ctx] - best_exp_end) 
				+ hd->beta*(local_data_penalty[worker_ctx])
				+ hd->_gamma*(local_power[worker_ctx]);

		if (exp_end[worker_ctx] > max_exp_end)
			/* This placement will make the computation
			 * longer, take into account the idle
			 * consumption of other cpus */
			fitness[worker_ctx] += hd->_gamma * hd->idle_power * (exp_end[worker_ctx] - max_exp_end) / 1000000.0;

		if (best == -1 || fitness[worker_ctx] < best_fitness)
		{
			/* we found a better solution */
			best_fitness = fitness[worker_ctx];
			best = worker;
			best_id_ctx = worker_ctx;
		}
		worker_ctx++;
	}

	/* By now, we must have found a solution */
	STARPU_ASSERT(best != -1);
	
	/* we should now have the best worker in variable "best" */
	double model_best;

	if (bundle)
	{
		/* If we have a task bundle, we have computed the expected
		 * length for the entire bundle, but not for the task alone. */
		enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(best);
		model_best = starpu_task_expected_length(task, perf_arch,
				_starpu_get_job_associated_to_task(task)->nimpl);

		/* Remove the task from the bundle since we have made a
		 * decision for it, and that other tasks should not consider it
		 * anymore. */
		PTHREAD_MUTEX_LOCK(&bundle->mutex);
		int ret = starpu_task_bundle_remove(bundle, task);
		
		/* Perhaps the bundle was destroyed when removing the last
		 * entry */
		if (ret != 1)
			PTHREAD_MUTEX_UNLOCK(&bundle->mutex);
	}
	else 
	{
		model_best = local_task_length[best_id_ctx];
	}

	if(workers->init_cursor)
		workers->deinit_cursor(workers);

	return push_task_on_best_worker(task, best, model_best, prio, sched_ctx_id);
}

static int heft_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	pthread_mutex_t *changing_ctx_mutex = starpu_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers; 
	int ret_val = -1;
	if (task->priority > 0)
	{
		PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
		nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
		if(nworkers == 0)
		{
			PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
			return ret_val;
		}

		ret_val = _heft_push_task(task, 1, sched_ctx_id);
		PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
	nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
	if(nworkers == 0)
	{
		PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
		return ret_val;
	}

	ret_val = _heft_push_task(task, 0, sched_ctx_id);
	PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
	return ret_val;
}

static void heft_deinit(unsigned sched_ctx_id) 
{
	heft_data *ht = (heft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	free(ht);
}

struct starpu_sched_policy_s heft_policy = {
	.init_sched = heft_init,
	.deinit_sched = heft_deinit,
	.push_task = heft_push_task, 
	.push_task_notify = heft_push_task_notify,
	.pop_task = NULL,
	.pop_every_task = NULL,
	.post_exec_hook = heft_post_exec_hook,
	.add_workers = heft_add_workers	,
	.remove_workers = heft_remove_workers,
	.policy_name = "heft",
	.policy_description = "Heterogeneous Earliest Finish Task"
};
