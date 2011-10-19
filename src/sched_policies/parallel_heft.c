/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
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
#include <core/perfmodel/perfmodel.h>
#include <starpu_parameters.h>
#include <common/barrier.h>

static pthread_mutex_t big_lock;

static unsigned  ncombinedworkers;
//static enum starpu_perf_archtype applicable_perf_archtypes[STARPU_NARCH_VARIATIONS];
//static unsigned napplicable_perf_archtypes = 0;

typedef struct {
	double alpha;
	double beta;
	double _gamma;
	double idle_power;
} pheft_data;

static double worker_exp_start[STARPU_NMAXWORKERS];
static double worker_exp_end[STARPU_NMAXWORKERS];
static double worker_exp_len[STARPU_NMAXWORKERS];
static int ntasks[STARPU_NMAXWORKERS];

static void parallel_heft_post_exec_hook(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (!task->cl || task->execute_on_a_specific_worker)
		return;

	int workerid = starpu_worker_get_id();
	double model = task->predicted;
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	if (model < 0.0)
		model = 0.0;
	
	/* Once we have executed the task, we can update the predicted amount
	 * of work. */
	PTHREAD_MUTEX_LOCK(sched_ctx->sched_mutex[workerid]);
	worker_exp_len[workerid] -= model;
	worker_exp_start[workerid] = starpu_timing_now();
	worker_exp_end[workerid] = worker_exp_start[workerid] + worker_exp_len[workerid];
	ntasks[workerid]--;
	PTHREAD_MUTEX_UNLOCK(sched_ctx->sched_mutex[workerid]);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double exp_end_predicted, int prio, struct starpu_sched_ctx *sched_ctx)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	/* Is this a basic worker or a combined worker ? */
//	int nbasic_workers = (int)starpu_worker_get_count();
	int nbasic_workers = sched_ctx->nworkers;
	int is_basic_worker = (best_workerid < nbasic_workers);

	unsigned memory_node; 
	memory_node = starpu_worker_get_memory_node(best_workerid);

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_on_node(task, memory_node);

	int ret = 0;

	PTHREAD_MUTEX_LOCK(&big_lock);

	if (is_basic_worker)
	{
		task->predicted = exp_end_predicted - worker_exp_end[best_workerid];
		worker_exp_len[best_workerid] += exp_end_predicted - worker_exp_end[best_workerid];
		worker_exp_end[best_workerid] = exp_end_predicted;
		worker_exp_start[best_workerid] = exp_end_predicted - worker_exp_len[best_workerid];
	
		ntasks[best_workerid]++;

		ret = starpu_push_local_task(best_workerid, task, prio);
	}
	else {
		/* This is a combined worker so we create task aliases */
		struct starpu_combined_worker_s *combined_worker;
		combined_worker = _starpu_get_combined_worker_struct(best_workerid);
		int worker_size = combined_worker->worker_size;
		int *combined_workerid = combined_worker->combined_workerid;

		starpu_job_t j = _starpu_get_job_associated_to_task(task);
		j->task_size = worker_size;
		j->combined_workerid = best_workerid;
		j->active_task_alias_count = 0;

		PTHREAD_BARRIER_INIT(&j->before_work_barrier, NULL, worker_size);
		PTHREAD_BARRIER_INIT(&j->after_work_barrier, NULL, worker_size);

		int i;
		for (i = 0; i < worker_size; i++)
		{
			struct starpu_task *alias = _starpu_create_task_alias(task);
			int local_worker = combined_workerid[i];

			alias->predicted = exp_end_predicted - worker_exp_end[local_worker];
	
			worker_exp_len[local_worker] += exp_end_predicted - worker_exp_end[local_worker];
			worker_exp_end[local_worker] = exp_end_predicted;
			worker_exp_start[local_worker] = exp_end_predicted - worker_exp_len[local_worker];
		
			ntasks[local_worker]++;
	
			ret |= starpu_push_local_task(local_worker, alias, prio);
		}

	}

	PTHREAD_MUTEX_UNLOCK(&big_lock);

	return ret;
}

static double compute_expected_end(int workerid, double length, int nworkers)
{
	if (workerid < (int)nworkers)
	{
		/* This is a basic worker */
		return worker_exp_start[workerid] + worker_exp_len[workerid] + length;
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
			double local_exp_start = worker_exp_start[combined_workerid[i]];
			double local_exp_len = worker_exp_len[combined_workerid[i]];
			double local_exp_end = local_exp_start + local_exp_len + length;
			exp_end = STARPU_MAX(exp_end, local_exp_end);
		}

		return exp_end;
	}
}

static double compute_ntasks_end(int workerid, int nworkers)
{
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	if (workerid < (int)nworkers)
	{
		/* This is a basic worker */
		return ntasks[workerid] / starpu_worker_get_relative_speedup(perf_arch);
	}
	else {
		/* This is a combined worker, the expected end is the end for the latest worker */
		int worker_size;
		int *combined_workerid;
		starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

		int ntasks_end=0;

		int i;
		for (i = 0; i < worker_size; i++)
		{
			/* XXX: this is actually bogus: not all pushed tasks are necessarily parallel... */
			ntasks_end = STARPU_MAX(ntasks_end, ntasks[combined_workerid[i]] / starpu_worker_get_relative_speedup(perf_arch));
		}

		return ntasks_end;
	}
}

static int _parallel_heft_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);

	pheft_data *hd = (pheft_data*)sched_ctx->policy_data;
	unsigned nworkers_ctx = sched_ctx->nworkers;
	unsigned worker, worker_ctx;
	int best = -1, best_id_ctx = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1, forced_best_ctx = -1;

	double local_task_length[nworkers_ctx + ncombinedworkers];
	double local_data_penalty[nworkers_ctx + ncombinedworkers];
	double local_power[nworkers_ctx + ncombinedworkers];
	double local_exp_end[nworkers_ctx + ncombinedworkers];
	double fitness[nworkers_ctx + ncombinedworkers];

	double max_exp_end = 0.0;

	int skip_worker[nworkers_ctx + ncombinedworkers];

	double best_exp_end = DBL_MAX;
	//double penality_best = 0.0;

	int ntasks_best = -1, ntasks_best_ctx = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;

	/* A priori, we know all estimations */
	int unknown = 0;

	for (worker_ctx = 0; worker_ctx < nworkers_ctx; worker_ctx++)
	{
		worker = sched_ctx->workerids[worker_ctx];
		/* Sometimes workers didn't take the tasks as early as we expected */
		worker_exp_start[worker] = STARPU_MAX(worker_exp_start[worker], starpu_timing_now());
		worker_exp_end[worker] = worker_exp_start[worker] + worker_exp_len[worker];
		if (worker_exp_end[worker] > max_exp_end)
			max_exp_end = worker_exp_end[worker];
	}

	unsigned nimpl;
	unsigned best_impl = 0;
	for (worker_ctx = 0; worker_ctx < (nworkers_ctx + ncombinedworkers); worker_ctx++)
 	{
		worker = sched_ctx->workerids[worker_ctx];
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!starpu_combined_worker_may_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				skip_worker[worker] = 1;
				continue;
			}
			else {
				skip_worker[worker] = 0;
			}

			enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);

			local_task_length[worker_ctx] = starpu_task_expected_length(task, perf_arch,nimpl);

			unsigned memory_node = starpu_worker_get_memory_node(worker);
			local_data_penalty[worker_ctx] = starpu_task_expected_data_transfer_time(memory_node, task);

			double ntasks_end = compute_ntasks_end(worker, nworkers_ctx);

			if (ntasks_best == -1
					|| (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
					|| (!calibrating && local_task_length[worker] == -1.0) /* Not calibrating but this worker is being calibrated */
					|| (calibrating && local_task_length[worker] == -1.0 && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
					) {
				ntasks_best_end = ntasks_end;
				ntasks_best = worker;
				ntasks_best_ctx = worker_ctx;
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

			local_exp_end[worker_ctx] = compute_expected_end(worker, local_task_length[worker], nworkers_ctx);

			//fprintf(stderr, "WORKER %d -> length %e end %e\n", worker, local_task_length[worker], local_exp_end[worker]);

			if (local_exp_end[worker_ctx] < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = local_exp_end[worker_ctx];
				best_impl = nimpl;
			}


			local_power[worker_ctx] = starpu_task_expected_power(task, perf_arch,nimpl);
			//_STARPU_DEBUG("Scheduler parallel heft: task length (%lf) local power (%lf) worker (%u) kernel (%u) \n", local_task_length[worker],local_power[worker],worker,nimpl);

			if (local_power[worker_ctx] == -1.0)
				local_power[worker_ctx] = 0.;

		} //end for
	}

	if (unknown)
	{
		forced_best = ntasks_best;
		forced_best_ctx = ntasks_best_ctx;
	}


	double best_fitness = -1;


	if (forced_best == -1)
	{
		for (worker_ctx = 0; worker_ctx < nworkers_ctx + ncombinedworkers; worker_ctx++)
		{
			/* if combinedworker don't search the id in the ctx */
			worker = worker_ctx >= nworkers_ctx ? worker_ctx : 
				sched_ctx->workerids[worker_ctx];

			if (skip_worker[worker_ctx])
			{
				/* no one on that queue may execute this task */
				continue;
			}
	
			fitness[worker_ctx] = hd->alpha*(local_exp_end[worker_ctx] - best_exp_end) 
					+ hd->beta*(local_data_penalty[worker_ctx])
					+ hd->_gamma*(local_power[worker_ctx]);

			if (local_exp_end[worker_ctx] > max_exp_end)
				/* This placement will make the computation
				 * longer, take into account the idle
				 * consumption of other cpus */
				fitness[worker_ctx] += hd->_gamma * hd->idle_power * (local_exp_end[worker_ctx] - max_exp_end) / 1000000.0;

			if (best == -1 || fitness[worker_ctx] < best_fitness)
			{
				/* we found a better solution */
				best_fitness = fitness[worker_ctx];
				best = worker;
				best_id_ctx = worker_ctx;
			}

		//	fprintf(stderr, "FITNESS worker %d -> %e local_exp_end %e - local_data_penalty %e\n", worker, fitness[worker], local_exp_end[worker] - best_exp_end, local_data_penalty[worker]);
		}
	}

	STARPU_ASSERT(forced_best != -1 || best != -1);

	if (forced_best != -1)
	{
		/* there is no prediction available for that task
		 * with that arch we want to speed-up calibration time
		 * so we force this measurement */
		best = forced_best;
		best_id_ctx = forced_best_ctx;
		//penality_best = 0.0;
		best_exp_end = local_exp_end[best_id_ctx];
	}
	else 
	{
                //penality_best = local_data_penalty[best];
		best_exp_end = local_exp_end[best_id_ctx];
	}


	//_STARPU_DEBUG("Scheduler parallel heft: kernel (%u)\n", best_impl);
	_starpu_get_job_associated_to_task(task)->nimpl = best_impl;
	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, best_exp_end, prio, sched_ctx);
}

static int parallel_heft_push_task(struct starpu_task *task, unsigned sched_ctx_id)
{
	if (task->priority == STARPU_MAX_PRIO)
		return _parallel_heft_push_task(task, 1, sched_ctx_id);

	return _parallel_heft_push_task(task, 0, sched_ctx_id);
}

static void parallel_heft_init_for_workers(unsigned sched_ctx_id, int *workerids, unsigned nnew_workers)
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	
	int workerid;
	unsigned i;
	for (i = 0; i < nnew_workers; i++)
	{
		workerid = workerids[i];
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		/* init these structures only once for each worker */
		if(!workerarg->has_prev_init)
		{
			worker_exp_start[workerid] = starpu_timing_now();
			worker_exp_len[workerid] = 0.0;
			worker_exp_end[workerid] = worker_exp_start[workerid]; 
			ntasks[workerid] = 0;
			workerarg->has_prev_init = 1;
		}

		PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid], NULL);
	}
}

static void initialize_parallel_heft_policy(unsigned sched_ctx_id) 
{	
	pheft_data *hd = (pheft_data*)malloc(sizeof(pheft_data));
	hd->alpha = STARPU_DEFAULT_ALPHA;
	hd->beta = STARPU_DEFAULT_BETA;
	hd->_gamma = STARPU_DEFAULT_GAMMA;
	hd->idle_power = 0.0;
	
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	unsigned nworkers_ctx = sched_ctx->nworkers;
	sched_ctx->policy_data = (void*)hd;

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

	struct starpu_machine_config_s *config = (struct starpu_machine_config_s *)_starpu_get_machine_config();
	_starpu_sched_find_worker_combinations(&config->topology);

	ncombinedworkers = config->topology.ncombinedworkers;

	unsigned workerid_ctx;
	int workerid;
	for (workerid_ctx = 0; workerid_ctx < nworkers_ctx; workerid_ctx++)
	{
		workerid = sched_ctx->workerids[workerid_ctx];
		struct starpu_worker_s *workerarg = _starpu_get_worker_struct(workerid);
		if(!workerarg->has_prev_init)
		{
			worker_exp_start[workerid] = starpu_timing_now();
			worker_exp_len[workerid] = 0.0;
			worker_exp_end[workerid] = worker_exp_start[workerid]; 
			ntasks[workerid] = 0;
		}
		PTHREAD_MUTEX_INIT(sched_ctx->sched_mutex[workerid], NULL);
		PTHREAD_COND_INIT(sched_ctx->sched_cond[workerid], NULL);
	}

	PTHREAD_MUTEX_INIT(&big_lock, NULL);

	/* We pre-compute an array of all the perfmodel archs that are applicable */
	unsigned total_worker_count = nworkers_ctx + ncombinedworkers;

	unsigned used_perf_archtypes[STARPU_NARCH_VARIATIONS];
	memset(used_perf_archtypes, 0, sizeof(used_perf_archtypes));

	for (workerid = 0; workerid < total_worker_count; workerid++)
	{
		enum starpu_perf_archtype perf_archtype = starpu_worker_get_perf_archtype(workerid);
		used_perf_archtypes[perf_archtype] = 1;
	}

//	napplicable_perf_archtypes = 0;

//	int arch;
//	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++)
//	{
//		if (used_perf_archtypes[arch])
//			applicable_perf_archtypes[napplicable_perf_archtypes++] = arch;
//	}
}

static void parallel_heft_deinit(unsigned sched_ctx_id) 
{
	struct starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(sched_ctx_id);
	pheft_data *ht = (pheft_data*)sched_ctx->policy_data;	  
	unsigned workerid_ctx;
	int workerid;

	unsigned nworkers_ctx = sched_ctx->nworkers;
	for (workerid_ctx = 0; workerid_ctx < nworkers_ctx; workerid_ctx++)
	{	
		workerid = sched_ctx->workerids[workerid_ctx];
		PTHREAD_MUTEX_DESTROY(sched_ctx->sched_mutex[workerid]);
		PTHREAD_COND_DESTROY(sched_ctx->sched_cond[workerid]);
	}

	free(ht);
	PTHREAD_MUTEX_DESTROY(&big_lock);
}

/* TODO: use post_exec_hook to fix the expected start */
struct starpu_sched_policy_s _starpu_sched_parallel_heft_policy = {
	.init_sched = initialize_parallel_heft_policy,
	.init_sched_for_workers = parallel_heft_init_for_workers,
	.deinit_sched = parallel_heft_deinit,
	.push_task = parallel_heft_push_task, 
	.pop_task = NULL,
	.post_exec_hook = parallel_heft_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "pheft",
	.policy_description = "parallel HEFT"
};
