/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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
#include <core/detect_combined_workers.h>
#include <core/sched_policy.h>
#include <core/task.h>

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

/* if no priority is set when creating the scheduling context, we use the following ones */
#define DEFAULT_MIN_PRIORITY 0
#define DEFAULT_MAX_PRIORITY 1

//static unsigned ncombinedworkers;
//static enum starpu_perfmodel_archtype applicable_perf_archtypes[STARPU_NARCH_VARIATIONS];
//static unsigned napplicable_perf_archtypes = 0;

/*
 * Here are the default values of alpha, beta, gamma
 */

#define _STARPU_SCHED_ALPHA_DEFAULT 1.0
#define _STARPU_SCHED_BETA_DEFAULT 1.0
#define _STARPU_SCHED_GAMMA_DEFAULT 1000.0

struct _starpu_pheft_data
{
	double alpha;
	double beta;
	double _gamma;
	double idle_power;
/* When we push a task on a combined worker we need all the cpu workers it contains
 * to be locked at once */
	starpu_pthread_mutex_t global_push_mutex;
};

static double worker_exp_start[STARPU_NMAXWORKERS];
static double worker_exp_end[STARPU_NMAXWORKERS];
static double worker_exp_len[STARPU_NMAXWORKERS];
static int ntasks[STARPU_NMAXWORKERS];


/*!!!!!!! It doesn't work with several contexts because the combined workers are constructed
  from the workers available to the program, and not to the context !!!!!!!!!!!!!!!!!!!!!!!
*/

static void parallel_heft_pre_exec_hook(struct starpu_task *task, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	if (!task->cl || task->execute_on_a_specific_worker)
		return;

	unsigned workerid = starpu_worker_get_id_check();
	double model = task->predicted;
	double transfer_model = task->predicted_transfer;
	const double now = starpu_timing_now();

	if (isnan(model))
		model = 0.0;

	if (isnan(transfer_model))
		transfer_model = 0.0;

	/* Once we have started the task, we can update the predicted amount
	 * of work. */
	starpu_worker_lock_self();
	worker_exp_len[workerid] -= model + transfer_model;
	worker_exp_start[workerid] = now + model;
	worker_exp_end[workerid] = worker_exp_start[workerid] + worker_exp_len[workerid];
	ntasks[workerid]--;
	starpu_worker_unlock_self();
}

static void parallel_heft_post_exec_hook(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	unsigned workerid = starpu_worker_get_id_check();
	const double now = starpu_timing_now();

	/* Once we have executed the task, we can update the predicted amount
	 * of work. */
	starpu_worker_lock_self();
	worker_exp_start[workerid] = now;
	worker_exp_end[workerid] = worker_exp_start[workerid] + worker_exp_len[workerid];
	starpu_worker_unlock_self();
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double exp_start_predicted, double exp_end_predicted, int prio, unsigned sched_ctx_id)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	struct _starpu_pheft_data *hd = (struct _starpu_pheft_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_for(task, best_workerid);

	int ret = 0;

	if (!starpu_worker_is_combined_worker(best_workerid))
	{
		starpu_worker_lock(best_workerid);
		task->predicted = exp_end_predicted - exp_start_predicted;
		/* TODO */
		task->predicted_transfer = 0;
		worker_exp_len[best_workerid] += task->predicted;
		worker_exp_end[best_workerid] = exp_end_predicted;
		worker_exp_start[best_workerid] = exp_end_predicted - worker_exp_len[best_workerid];

		ntasks[best_workerid]++;
		starpu_worker_unlock(best_workerid);

		/* We don't want it to interlace its task with a combined
		 * worker's one */
		starpu_worker_relax_on();
		STARPU_PTHREAD_MUTEX_LOCK(&hd->global_push_mutex);
		starpu_worker_relax_off();

		ret = starpu_push_local_task(best_workerid, task, prio);

		STARPU_PTHREAD_MUTEX_UNLOCK(&hd->global_push_mutex);
	}
	else
	{
		/* This task doesn't belong to an actual worker, it belongs
		 * to a combined worker and thus the scheduler doesn't care
		 * of its predicted values which are insignificant */
		task->predicted = 0;
		task->predicted_transfer = 0;

		starpu_parallel_task_barrier_init(task, best_workerid);
		int worker_size = 0;
		int *combined_workerid;
		starpu_combined_worker_get_description(best_workerid, &worker_size, &combined_workerid);

		/* All cpu workers must be locked at once */
		starpu_worker_relax_on();
		STARPU_PTHREAD_MUTEX_LOCK(&hd->global_push_mutex);
		starpu_worker_relax_off();

		/* This is a combined worker so we create task aliases */
		int i;
		for (i = 0; i < worker_size; i++)
		{
			struct starpu_task *alias = starpu_task_dup(task);
			int local_combined_workerid = combined_workerid[i];

			alias->predicted = exp_end_predicted - worker_exp_end[local_combined_workerid];
			/* TODO */
			alias->predicted_transfer = 0;
			alias->destroy = 1;
			starpu_worker_lock(local_combined_workerid);
			worker_exp_len[local_combined_workerid] += alias->predicted;
			worker_exp_end[local_combined_workerid] = exp_end_predicted;
			worker_exp_start[local_combined_workerid] = exp_end_predicted - worker_exp_len[local_combined_workerid];

			ntasks[local_combined_workerid]++;
			starpu_worker_unlock(local_combined_workerid);

			_STARPU_TRACE_JOB_PUSH(alias, alias->priority > 0);
			ret |= starpu_push_local_task(local_combined_workerid, alias, prio);
		}

		STARPU_PTHREAD_MUTEX_UNLOCK(&hd->global_push_mutex);

	}

	return ret;
}

static double compute_expected_end(double *_worker_exp_end, int workerid, double length)
{
	if (!starpu_worker_is_combined_worker(workerid))
	{
		double res;
		/* This is a basic worker */

		res = _worker_exp_end[workerid] + length;

		return res;
	}
	else
	{
		/* This is a combined worker, the expected end is the end for the latest worker */
		int worker_size;
		int *combined_workerid;
		starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

		double exp_end = DBL_MIN;

		int i;
		for (i = 0; i < worker_size; i++)
		{
			double local_exp_end = _worker_exp_end[combined_workerid[i]] + length;
			exp_end = STARPU_MAX(exp_end, local_exp_end);
		}

		return exp_end;
	}
}

static double compute_ntasks_end(int workerid, unsigned sched_ctx_id)
{
	struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerid, sched_ctx_id);

	if (!starpu_worker_is_combined_worker(workerid))
	{
		double res;
		/* This is a basic worker */

		/* Here helgrind would shout that this is unprotected, but we
		 * are fine with getting outdated values, this is just an
		 * estimation */
		res = ntasks[workerid] / starpu_worker_get_relative_speedup(perf_arch);

		return res;
	}
	else
	{
		/* This is a combined worker, the expected end is the end for the latest worker */
		int worker_size;
		int *combined_workerid;
		starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

		int ntasks_end=0;

		/* Here helgrind would shout that this is unprotected, but we
		 * are fine with getting outdated values, this is just an
		 * estimation */
		int i;
		for (i = 0; i < worker_size; i++)
		{
			/* XXX: this is actually bogus: not all pushed tasks are necessarily parallel... */
			ntasks_end = STARPU_MAX(ntasks_end, (int) ((double) ntasks[combined_workerid[i]] / starpu_worker_get_relative_speedup(perf_arch)));
		}

		return ntasks_end;
	}
}

static int _parallel_heft_push_task(struct starpu_task *task, unsigned prio, unsigned sched_ctx_id)
{
	struct _starpu_pheft_data *hd = (struct _starpu_pheft_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	unsigned nworkers_ctx = workers->nworkers;

	unsigned workerid, worker_ctx = 0;
	int best = -1, best_id_ctx = -1;

	/* this flag is set if the corresponding workerid is selected because
	   there is no performance prediction available yet */
	int forced_best = -1, forced_best_ctx = -1, forced_nimpl = -1;

	double local_task_length[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];
	double local_data_penalty[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];
	double local_energy[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];
	double local_exp_end[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];
	double fitness[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];

	double max_exp_end = 0.0;

	int skip_worker[nworkers_ctx][STARPU_MAXIMPLEMENTATIONS];

	double best_exp_start;
	double best_exp_end = DBL_MAX;
	//double penality_best = 0.0;

	int ntasks_best = -1, ntasks_best_ctx = -1, nimpl_best = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;

	/* A priori, we know all estimations */
	int unknown = 0;
	struct starpu_sched_ctx_iterator it;

	double now = starpu_timing_now();
	double _worker_exp_end[nworkers_ctx];

	memset(skip_worker, 0, nworkers_ctx*STARPU_MAXIMPLEMENTATIONS*sizeof(int));

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		workerid = workers->get_next(workers, &it);

		if(!starpu_worker_is_combined_worker(workerid))
		{
			/* Here helgrind would shout that this is unprotected, but we
			 * are fine with getting outdated values, this is just an
			 * estimation */
			/* Sometimes workers didn't take the tasks as early as we expected */
			double exp_start = STARPU_MAX(worker_exp_start[workerid], now);
			_worker_exp_end[workerid] = exp_start + worker_exp_len[workerid];
			if (_worker_exp_end[workerid] > max_exp_end)
				max_exp_end = _worker_exp_end[workerid];
		}
	}

	unsigned nimpl;
	worker_ctx = 0;
	while(workers->has_next(workers, &it))
	{
                workerid = workers->get_next(workers, &it);

		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				/* no one on that queue may execute this task */
				skip_worker[worker_ctx][nimpl] = 1;
				continue;
			}
			else
			{
				skip_worker[worker_ctx][nimpl] = 0;
			}


			struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(workerid, sched_ctx_id);

			local_task_length[worker_ctx][nimpl] = starpu_task_expected_length(task, perf_arch,nimpl);

			local_data_penalty[worker_ctx][nimpl] = starpu_task_expected_data_transfer_time_for(task, workerid);

			double ntasks_end = compute_ntasks_end(workerid, sched_ctx_id);

			if (ntasks_best == -1
			    || (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
			    || (!calibrating && isnan(local_task_length[worker_ctx][nimpl])) /* Not calibrating but this workerid is being calibrated */
			    || (calibrating && isnan(local_task_length[worker_ctx][nimpl]) && ntasks_end < ntasks_best_end) /* Calibrating, compete this workerid with other non-calibrated */
					)
			{
				ntasks_best_end = ntasks_end;
				ntasks_best = workerid;
				ntasks_best_ctx = worker_ctx;
				nimpl_best = nimpl;
			}

			if (isnan(local_task_length[worker_ctx][nimpl]))
			{
				static int warned;
				if (!warned)
				{
					warned = 1;
					_STARPU_DISP("Warning: performance model for %s not finished calibrating on %u, using a dumb scheduling heuristic for now\n", starpu_task_get_name(task), workerid);
				}
				/* we are calibrating, we want to speed-up calibration time
				 * so we privilege non-calibrated tasks (but still
				 * greedily distribute them to avoid dumb schedules) */
				calibrating = 1;
			}

			if (isnan(local_task_length[worker_ctx][nimpl])
					|| _STARPU_IS_ZERO(local_task_length[worker_ctx][nimpl]))
				/* there is no prediction available for that task
				 * with that arch yet, so switch to a greedy strategy */
				unknown = 1;

			if (unknown)
				continue;

			local_exp_end[worker_ctx][nimpl] = compute_expected_end(_worker_exp_end, workerid, local_task_length[worker_ctx][nimpl]);

			//fprintf(stderr, "WORKER %d -> length %e end %e\n", workerid, local_task_length[worker_ctx][nimpl], local_exp_end[workerid][nimpl]);

			if (local_exp_end[worker_ctx][nimpl] < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = local_exp_end[worker_ctx][nimpl];
				nimpl_best = nimpl;
			}


			local_energy[worker_ctx][nimpl] = starpu_task_expected_energy(task, perf_arch,nimpl);
			//_STARPU_DEBUG("Scheduler parallel heft: task length (%lf) local energy (%lf) workerid (%u) kernel (%u) \n", local_task_length[workerid],local_energy[workerid],workerid,nimpl);

			if (isnan(local_energy[worker_ctx][nimpl]))
				local_energy[worker_ctx][nimpl] = 0.;

		}
		worker_ctx++;
	}

	if (unknown)
	{
		forced_best = ntasks_best;
		forced_best_ctx = ntasks_best_ctx;
		forced_nimpl = nimpl_best;
	}


	if (forced_best == -1)
	{
		double best_fitness = -1;
		worker_ctx = 0;
		while(workers->has_next(workers, &it))
		{
			workerid = workers->get_next(workers, &it);

			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (skip_worker[worker_ctx][nimpl])
				{
					/* no one on that queue may execute this task */
					continue;
				}

				fitness[worker_ctx][nimpl] = hd->alpha*(local_exp_end[worker_ctx][nimpl] - best_exp_end)
						+ hd->beta*(local_data_penalty[worker_ctx][nimpl])
						+ hd->_gamma*(local_energy[worker_ctx][nimpl]);

				if (local_exp_end[worker_ctx][nimpl] > max_exp_end)
					/* This placement will make the computation
					 * longer, take into account the idle
					 * consumption of other cpus */
					fitness[worker_ctx][nimpl] += hd->_gamma * hd->idle_power * (local_exp_end[worker_ctx][nimpl] - max_exp_end) / 1000000.0;

				if (best == -1 || fitness[worker_ctx][nimpl] < best_fitness)
				{
					/* we found a better solution */
					best_fitness = fitness[worker_ctx][nimpl];
					best = workerid;
					best_id_ctx = worker_ctx;
					nimpl_best = nimpl;
				}

			//	fprintf(stderr, "FITNESS workerid %d -> %e local_exp_end %e - local_data_penalty %e\n", workerid, fitness[workerid][nimpl], local_exp_end[workerid][nimpl] - best_exp_end, local_data_penalty[workerid][nimpl]);
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
		best_id_ctx = forced_best_ctx;
		nimpl_best = forced_nimpl;
		//penality_best = 0.0;
		best_exp_end = compute_expected_end(_worker_exp_end, best, 0);
	}
	else
	{
		//penality_best = local_data_penalty[best_id_ctx][nimpl_best];
		best_exp_end = local_exp_end[best_id_ctx][nimpl_best];
	}
	best_exp_start = _worker_exp_end[best];

	//_STARPU_DEBUG("Scheduler parallel heft: kernel (%u)\n", nimpl_best);
	starpu_task_set_implementation(task, nimpl_best);
	/* we should now have the best workerid in variable "best" */
	starpu_sched_task_break(task);
	return push_task_on_best_worker(task, best, best_exp_start, best_exp_end, prio, sched_ctx_id);
}

static int parallel_heft_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	int ret_val = -1;

	if (task->priority == STARPU_MAX_PRIO)
	{
		ret_val = _parallel_heft_push_task(task, 1, sched_ctx_id);
                return ret_val;
        }

        ret_val = _parallel_heft_push_task(task, 0, sched_ctx_id);
	return ret_val;
}

static void parallel_heft_add_workers(__attribute__((unused)) unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	unsigned i;
	double now = starpu_timing_now();
	for (i = 0; i < nworkers; i++)
	{
		int workerid = workerids[i];
		struct _starpu_worker *workerarg = _starpu_get_worker_struct(workerid);
		/* init these structures only once for each worker */
		if(!workerarg->has_prev_init)
		{
			worker_exp_start[workerid] = now;
			worker_exp_len[workerid] = 0.0;
			worker_exp_end[workerid] = worker_exp_start[workerid];
			ntasks[workerid] = 0;
			workerarg->has_prev_init = 1;
		}
	}
	_starpu_sched_find_worker_combinations(workerids, nworkers);

// start_unclear_part: not very clear where this is used
/* 	struct _starpu_machine_config *config = _starpu_get_machine_config(); */
/* 	ncombinedworkers = config->topology.ncombinedworkers; */

/* 	/\* We pre-compute an array of all the perfmodel archs that are applicable *\/ */
/* 	unsigned total_worker_count = nworkers + ncombinedworkers; */

/* 	unsigned used_perf_archtypes[STARPU_NARCH_VARIATIONS]; */
/* 	memset(used_perf_archtypes, 0, sizeof(used_perf_archtypes)); */

/* 	for (workerid = 0; workerid < total_worker_count; workerid++) */
/* 	{ */
/* 		enum starpu_perfmodel_archtype perf_archtype = starpu_worker_get_perf_archtype(workerid); */
/* 		used_perf_archtypes[perf_archtype] = 1; */
/* 	} */

// end_unclear_part

//	napplicable_perf_archtypes = 0;

//	int arch;
//	for (arch = 0; arch < STARPU_NARCH_VARIATIONS; arch++)
//	{
//		if (used_perf_archtypes[arch])
//			applicable_perf_archtypes[napplicable_perf_archtypes++] = arch;
//	}

}

static void initialize_parallel_heft_policy(unsigned sched_ctx_id)
{
	struct _starpu_pheft_data *hd;
	_STARPU_MALLOC(hd, sizeof(struct _starpu_pheft_data));

	_STARPU_DISP("Warning: the pheft scheduler is mostly a proof of concept and not really very optimized\n");

	if (starpu_sched_ctx_min_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_min_priority(sched_ctx_id, DEFAULT_MIN_PRIORITY);
	if (starpu_sched_ctx_max_priority_is_set(sched_ctx_id) == 0)
		starpu_sched_ctx_set_max_priority(sched_ctx_id, DEFAULT_MAX_PRIORITY);
	STARPU_ASSERT_MSG(starpu_sched_ctx_get_min_priority(sched_ctx_id) < starpu_sched_ctx_get_max_priority(sched_ctx_id),
			  "Priority min %d should be lower than priority max %d\n",
			  starpu_sched_ctx_get_min_priority(sched_ctx_id), starpu_sched_ctx_get_max_priority(sched_ctx_id));

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)hd);

	hd->alpha = starpu_get_env_float_default("STARPU_SCHED_ALPHA", _STARPU_SCHED_ALPHA_DEFAULT);
	hd->beta = starpu_get_env_float_default("STARPU_SCHED_BETA", _STARPU_SCHED_BETA_DEFAULT);
#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (starpu_getenv("STARPU_SCHED_GAMMA"))
		_STARPU_DISP("Warning: STARPU_SCHED_GAMMA was used, but --enable-blocking-drivers configuration was not set, CPU cores will not actually be sleeping\n");
#endif
	hd->_gamma = starpu_get_env_float_default("STARPU_SCHED_GAMMA", _STARPU_SCHED_GAMMA_DEFAULT);
	hd->idle_power = starpu_get_env_float_default("STARPU_IDLE_POWER", 0.0);

	STARPU_PTHREAD_MUTEX_INIT(&hd->global_push_mutex, NULL);

	/* Tell helgrind that we are fine with getting outdated values when
	 * estimating schedules */
	STARPU_HG_DISABLE_CHECKING(worker_exp_start);
	STARPU_HG_DISABLE_CHECKING(worker_exp_end);
	STARPU_HG_DISABLE_CHECKING(worker_exp_len);
	STARPU_HG_DISABLE_CHECKING(ntasks);
}

static void parallel_heft_deinit(unsigned sched_ctx_id)
{
	struct _starpu_pheft_data *hd = (struct _starpu_pheft_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_PTHREAD_MUTEX_DESTROY(&hd->global_push_mutex);
	free(hd);
}

struct starpu_sched_policy _starpu_sched_parallel_heft_policy =
{
	.init_sched = initialize_parallel_heft_policy,
	.deinit_sched = parallel_heft_deinit,
	.add_workers = parallel_heft_add_workers,
	.remove_workers = NULL,
	.push_task = parallel_heft_push_task,
	.pop_task = NULL,
	.pre_exec_hook = parallel_heft_pre_exec_hook,
	.post_exec_hook = parallel_heft_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "pheft",
	.policy_description = "parallel HEFT",
	.worker_type = STARPU_WORKER_LIST,
};
