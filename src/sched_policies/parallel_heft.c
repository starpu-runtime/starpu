/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 inria
 * Copyright (C) 2010-2012  Université de Bordeaux 1
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
#include <sched_policies/detect_combined_workers.h>

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

static unsigned ncombinedworkers;
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


/*!!!!!!! It doesn't work with several contexts because the combined workers are constructed         
  from the workers available to the program, and not to the context !!!!!!!!!!!!!!!!!!!!!!!          
*/

static void parallel_heft_post_exec_hook(struct starpu_task *task)
{
	if (!task->cl || task->execute_on_a_specific_worker)
		return;

	int workerid = starpu_worker_get_id();
	double model = task->predicted;
	unsigned sched_ctx_id = task->sched_ctx;
	double transfer_model = task->predicted_transfer;

	if (isnan(model))
		model = 0.0;

	pthread_mutex_t *sched_mutex;
	pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(sched_ctx_id, workerid, &sched_mutex, &sched_cond);

	/* Once we have executed the task, we can update the predicted amount
	 * of work. */
	_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	worker_exp_len[workerid] -= model + transfer_model;
	worker_exp_start[workerid] = starpu_timing_now();
	worker_exp_end[workerid] = worker_exp_start[workerid] + worker_exp_len[workerid];
	ntasks[workerid]--;
	_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

static int push_task_on_best_worker(struct starpu_task *task, int best_workerid, double exp_end_predicted, int prio, unsigned sched_ctx_id)
{
	/* make sure someone coule execute that task ! */
	STARPU_ASSERT(best_workerid != -1);

	_starpu_increment_nsubmitted_tasks_of_worker(best_workerid);

	/* Is this a basic worker or a combined worker ? */
	int nbasic_workers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
	int is_basic_worker = (best_workerid < nbasic_workers);

	unsigned memory_node;
	memory_node = starpu_worker_get_memory_node(best_workerid);

	if (starpu_get_prefetch_flag())
		starpu_prefetch_task_input_on_node(task, memory_node);

	int ret = 0;

	if (is_basic_worker)
	{
		task->predicted = exp_end_predicted - worker_exp_end[best_workerid];
		/* TODO */
		task->predicted_transfer = 0;
		pthread_mutex_t *sched_mutex;
		pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(sched_ctx_id, best_workerid, &sched_mutex, &sched_cond);

		_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		worker_exp_len[best_workerid] += task->predicted;
		worker_exp_end[best_workerid] = exp_end_predicted;
		worker_exp_start[best_workerid] = exp_end_predicted - worker_exp_len[best_workerid];

		ntasks[best_workerid]++;
		_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

		ret = starpu_push_local_task(best_workerid, task, prio);
	}
	else
	{
		/* This is a combined worker so we create task aliases */
		struct _starpu_combined_worker *combined_worker;
		combined_worker = _starpu_get_combined_worker_struct(best_workerid);
		int worker_size = combined_worker->worker_size;
		int *combined_workerid = combined_worker->combined_workerid;

		struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
		j->task_size = worker_size;
		j->combined_workerid = best_workerid;
		j->active_task_alias_count = 0;
		task->predicted_transfer = 0;

		_STARPU_PTHREAD_BARRIER_INIT(&j->before_work_barrier, NULL, worker_size);
		_STARPU_PTHREAD_BARRIER_INIT(&j->after_work_barrier, NULL, worker_size);

		int i;
		for (i = 0; i < worker_size; i++)
		{
			struct starpu_task *alias = _starpu_create_task_alias(task);
			int local_worker = combined_workerid[i];

			alias->predicted = exp_end_predicted - worker_exp_end[local_worker];
			/* TODO */
			alias->predicted_transfer = 0;
			pthread_mutex_t *sched_mutex;
			pthread_cond_t *sched_cond;
			starpu_worker_get_sched_condition(sched_ctx_id, local_worker, &sched_mutex, &sched_cond);
			_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
			worker_exp_len[local_worker] += alias->predicted;
			worker_exp_end[local_worker] = exp_end_predicted;
			worker_exp_start[local_worker] = exp_end_predicted - worker_exp_len[local_worker];

			ntasks[local_worker]++;
			_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

			ret |= starpu_push_local_task(local_worker, alias, prio);
		}

	}

	return ret;
}

static double compute_expected_end(int workerid, double length, unsigned sched_ctx_id)
{
	unsigned nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
	if (workerid < (int)nworkers)
	{
		/* This is a basic worker */
		return worker_exp_start[workerid] + worker_exp_len[workerid] + length;
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
			double local_exp_start = worker_exp_start[combined_workerid[i]];
			double local_exp_len = worker_exp_len[combined_workerid[i]];
			double local_exp_end = local_exp_start + local_exp_len + length;
			exp_end = STARPU_MAX(exp_end, local_exp_end);
		}

		return exp_end;
	}
}

static double compute_ntasks_end(int workerid, unsigned sched_ctx_id)
{
	unsigned nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
	enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(workerid);
	if (workerid < (int)nworkers)
	{
		/* This is a basic worker */
		return ntasks[workerid] / starpu_worker_get_relative_speedup(perf_arch);
	}
	else
	{
		/* This is a combined worker, the expected end is the end for the latest worker */
		int worker_size;
		int *combined_workerid;
		starpu_combined_worker_get_description(workerid, &worker_size, &combined_workerid);

		int ntasks_end=0;

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
	pheft_data *hd = (pheft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	struct worker_collection *workers = starpu_get_worker_collection_of_sched_ctx(sched_ctx_id);
	unsigned nworkers_ctx = workers->nworkers;

	unsigned worker, worker_ctx = 0;
	int best = -1, best_id_ctx = -1;
	
	/* this flag is set if the corresponding worker is selected because
	   there is no performance prediction available yet */
	int forced_best = -1, forced_best_ctx = -1, forced_nimpl = -1;

	double local_task_length[nworkers_ctx + ncombinedworkers][STARPU_MAXIMPLEMENTATIONS];
	double local_data_penalty[nworkers_ctx + ncombinedworkers][STARPU_MAXIMPLEMENTATIONS];
	double local_power[nworkers_ctx + ncombinedworkers][STARPU_MAXIMPLEMENTATIONS];
	double local_exp_end[nworkers_ctx + ncombinedworkers][STARPU_MAXIMPLEMENTATIONS];
	double fitness[nworkers_ctx + ncombinedworkers][STARPU_MAXIMPLEMENTATIONS];

	double max_exp_end = 0.0;

	int skip_worker[nworkers_ctx + ncombinedworkers][STARPU_MAXIMPLEMENTATIONS];

	double best_exp_end = DBL_MAX;
	//double penality_best = 0.0;

	int ntasks_best = -1, ntasks_best_ctx = -1, nimpl_best = -1;
	double ntasks_best_end = 0.0;
	int calibrating = 0;

	/* A priori, we know all estimations */
	int unknown = 0;
	if(workers->init_cursor)
                workers->init_cursor(workers);

	while(workers->has_next(workers))
        {
                worker = workers->get_next(workers);

		pthread_mutex_t *sched_mutex;
		pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(sched_ctx_id, worker, &sched_mutex, &sched_cond);
		/* Sometimes workers didn't take the tasks as early as we expected */
		_STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
		worker_exp_start[worker] = STARPU_MAX(worker_exp_start[worker], starpu_timing_now());
		worker_exp_end[worker] = worker_exp_start[worker] + worker_exp_len[worker];
		if (worker_exp_end[worker] > max_exp_end)
			max_exp_end = worker_exp_end[worker];
		_STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
	}

	unsigned nimpl;
	while(workers->has_next(workers) || worker_ctx < (nworkers_ctx + ncombinedworkers))
	{
                worker = workers->has_next(workers) ? workers->get_next(workers) : worker_ctx;
		unsigned incremented = 0;
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if (!starpu_combined_worker_can_execute_task(worker, task, nimpl))
			{
				/* no one on that queue may execute this task */
				skip_worker[worker][nimpl] = 1;
				if(!incremented)
					worker_ctx++;
				continue;
			}
			else
			{
				skip_worker[worker][nimpl] = 0;
			}

			enum starpu_perf_archtype perf_arch = starpu_worker_get_perf_archtype(worker);

			local_task_length[worker_ctx][nimpl] = starpu_task_expected_length(task, perf_arch,nimpl);

			unsigned memory_node = starpu_worker_get_memory_node(worker);
			local_data_penalty[worker_ctx][nimpl] = starpu_task_expected_data_transfer_time(memory_node, task);

			double ntasks_end = compute_ntasks_end(worker, sched_ctx_id);

			if (ntasks_best == -1
			    || (!calibrating && ntasks_end < ntasks_best_end) /* Not calibrating, take better task */
			    || (!calibrating && isnan(local_task_length[worker_ctx][nimpl])) /* Not calibrating but this worker is being calibrated */
			    || (calibrating && isnan(local_task_length[worker_ctx][nimpl]) && ntasks_end < ntasks_best_end) /* Calibrating, compete this worker with other non-calibrated */
					)
			{
				ntasks_best_end = ntasks_end;
				ntasks_best = worker;
				ntasks_best_ctx = worker_ctx;
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
				 * with that arch yet, so switch to a greedy strategy */
				unknown = 1;

			if (unknown)
				continue;

			local_exp_end[worker_ctx][nimpl] = compute_expected_end(worker, local_task_length[worker_ctx][nimpl], sched_ctx_id);

			//fprintf(stderr, "WORKER %d -> length %e end %e\n", worker, local_task_length[worker_ctx][nimpl], local_exp_end[worker][nimpl]);

			if (local_exp_end[worker_ctx][nimpl] < best_exp_end)
			{
				/* a better solution was found */
				best_exp_end = local_exp_end[worker_ctx][nimpl];
				nimpl_best = nimpl;
			}


			local_power[worker_ctx][nimpl] = starpu_task_expected_power(task, perf_arch,nimpl);
			//_STARPU_DEBUG("Scheduler parallel heft: task length (%lf) local power (%lf) worker (%u) kernel (%u) \n", local_task_length[worker],local_power[worker],worker,nimpl);

			if (isnan(local_power[worker_ctx][nimpl]))
				local_power[worker_ctx][nimpl] = 0.;

		}
		if(!incremented)
			worker_ctx++;
	}

	if (unknown) {
		forced_best = ntasks_best;
		forced_best_ctx = ntasks_best_ctx;
		forced_nimpl = nimpl_best;
	}

	double best_fitness = -1;

	if (forced_best == -1)
	{
		worker_ctx = 0;
		while(workers->has_next(workers) || worker_ctx < (nworkers_ctx + ncombinedworkers))
		{
			worker = workers->has_next(workers) ? workers->get_next(workers) : worker_ctx;

			unsigned incremented = 0;
			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (skip_worker[worker_ctx][nimpl])
				{
					/* no one on that queue may execute this task */
					if(!incremented)
						worker_ctx++;

					incremented = 1;
					continue;
				}

				fitness[worker_ctx][nimpl] = hd->alpha*(local_exp_end[worker_ctx][nimpl] - best_exp_end) 
						+ hd->beta*(local_data_penalty[worker_ctx][nimpl])
						+ hd->_gamma*(local_power[worker_ctx][nimpl]);

				if (local_exp_end[worker_ctx][nimpl] > max_exp_end)
					/* This placement will make the computation
					 * longer, take into account the idle
					 * consumption of other cpus */
					fitness[worker_ctx][nimpl] += hd->_gamma * hd->idle_power * (local_exp_end[worker_ctx][nimpl] - max_exp_end) / 1000000.0;

				if (best == -1 || fitness[worker_ctx][nimpl] < best_fitness)
				{
					/* we found a better solution */
					best_fitness = fitness[worker_ctx][nimpl];
					best = worker;
					best_id_ctx = worker_ctx;
					nimpl_best = nimpl;
				}

			//	fprintf(stderr, "FITNESS worker %d -> %e local_exp_end %e - local_data_penalty %e\n", worker, fitness[worker][nimpl], local_exp_end[worker][nimpl] - best_exp_end, local_data_penalty[worker][nimpl]);
			}
			if(!incremented)
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
		best_exp_end = compute_expected_end(best, 0, sched_ctx_id);
	}
	else
	{
		//penality_best = local_data_penalty[best_id_ctx][nimpl_best];
		best_exp_end = local_exp_end[best_id_ctx][nimpl_best];
	}


	//_STARPU_DEBUG("Scheduler parallel heft: kernel (%u)\n", nimpl_best);
	_starpu_get_job_associated_to_task(task)->nimpl = nimpl_best;
	/* we should now have the best worker in variable "best" */
	return push_task_on_best_worker(task, best, best_exp_end, prio, sched_ctx_id);
}

static int parallel_heft_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	pthread_mutex_t *changing_ctx_mutex = starpu_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers;
	int ret_val = -1;

	if (task->priority == STARPU_MAX_PRIO)
	{  _STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
                nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
                if(nworkers == 0)
                {
                        _STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
                        return ret_val;
                }

		ret_val = _parallel_heft_push_task(task, 1, sched_ctx_id);
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

        ret_val = _parallel_heft_push_task(task, 0, sched_ctx_id);
	_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
	return ret_val;
}

static void parallel_heft_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	int workerid;
	unsigned i;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		struct _starpu_worker *workerarg = _starpu_get_worker_struct(workerid);
		/* init these structures only once for each worker */
		if(!workerarg->has_prev_init)
		{
			worker_exp_start[workerid] = starpu_timing_now();
			worker_exp_len[workerid] = 0.0;
			worker_exp_end[workerid] = worker_exp_start[workerid]; 
			ntasks[workerid] = 0;
			workerarg->has_prev_init = 1;
		}

		starpu_worker_init_sched_condition(sched_ctx_id, workerid);
	}

	struct _starpu_machine_config *config = (struct _starpu_machine_config *)_starpu_get_machine_config();
	_starpu_sched_find_worker_combinations(&config->topology);

	ncombinedworkers = config->topology.ncombinedworkers;

	/* We pre-compute an array of all the perfmodel archs that are applicable */
	unsigned total_worker_count = nworkers + ncombinedworkers;

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

static void parallel_heft_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	unsigned i;
	int worker;
	for(i = 0; i < nworkers; i++)
	{
		worker = workerids[i];
		starpu_worker_deinit_sched_condition(sched_ctx_id, worker);
	}
}
static void initialize_parallel_heft_policy(unsigned sched_ctx_id) 
{	
	starpu_create_worker_collection_for_sched_ctx(sched_ctx_id, WORKER_LIST);
	pheft_data *hd = (pheft_data*)malloc(sizeof(pheft_data));
	hd->alpha = _STARPU_DEFAULT_ALPHA;
	hd->beta = _STARPU_DEFAULT_BETA;
	hd->_gamma = _STARPU_DEFAULT_GAMMA;
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


}

static void parallel_heft_deinit(unsigned sched_ctx_id) 
{
	pheft_data *hd = (pheft_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	starpu_delete_worker_collection_for_sched_ctx(sched_ctx_id);
	free(hd);
}

/* TODO: use post_exec_hook to fix the expected start */
struct starpu_sched_policy _starpu_sched_parallel_heft_policy =
{
	.init_sched = initialize_parallel_heft_policy,
	.deinit_sched = parallel_heft_deinit,
	.add_workers = parallel_heft_add_workers,
	.remove_workers = parallel_heft_remove_workers,
	.push_task = parallel_heft_push_task, 
	.pop_task = NULL,
	.pre_exec_hook = NULL,
	.post_exec_hook = parallel_heft_post_exec_hook,
	.pop_every_task = NULL,
	.policy_name = "pheft",
	.policy_description = "parallel HEFT"
};
