/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <math.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <profiling/profiling.h>
#include <common/utils.h>
#include <core/debug.h>
#include <core/sched_ctx.h>
#include <drivers/driver_common/driver_common.h>
#include <starpu_top.h>
#include <core/sched_policy.h>
#include <top/starpu_top_core.h>
#include <core/debug.h>


#define BACKOFF_MAX 32  /* TODO : use parameter to define them */
#define BACKOFF_MIN 1

void _starpu_driver_start_job(struct _starpu_worker *worker, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch STARPU_ATTRIBUTE_UNUSED, int rank, int profiling)
{
	struct starpu_task *task = j->task;
	struct starpu_codelet *cl = task->cl;
	int starpu_top=_starpu_top_status_get();
	int workerid = worker->workerid;
	unsigned calibrate_model = 0;

	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	/* If the job is executed on a combined worker there is no need for the
	 * scheduler to process it : it doesn't contain any valuable data
	 * as it's not linked to an actual worker */
	if (j->task_size == 1 && rank == 0)
		_starpu_sched_pre_exec_hook(task);

	_starpu_set_worker_status(worker, STATUS_EXECUTING);

	if (rank == 0)
	{
		STARPU_ASSERT(task->status == STARPU_TASK_READY);
		task->status = STARPU_TASK_RUNNING;

		STARPU_AYU_RUNTASK(j->job_id);
		cl->per_worker_stats[workerid]++;

		struct starpu_profiling_task_info *profiling_info = task->profiling_info;

		if ((profiling && profiling_info) || calibrate_model || starpu_top)
		{
			_starpu_clock_gettime(&worker->cl_start);
			_starpu_worker_register_executing_start_date(workerid, &worker->cl_start);
		}
	}

	if (starpu_top)
		_starpu_top_task_started(task,workerid,&worker->cl_start);


	// Find out if the worker is the master of a parallel context
	struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(worker, j);
	if(!sched_ctx)
		sched_ctx = _starpu_get_sched_ctx_struct(j->task->sched_ctx);
	if(!sched_ctx->sched_policy)
	{
		if(!sched_ctx->awake_workers && sched_ctx->main_master == worker->workerid)
		{
			struct starpu_worker_collection *workers = sched_ctx->workers;
			struct starpu_sched_ctx_iterator it;
			int new_rank = 0;

			if (workers->init_iterator)
				workers->init_iterator(workers, &it);
			while (workers->has_next(workers, &it))
			{
				int _workerid = workers->get_next(workers, &it);
				if (_workerid != workerid)
				{
					new_rank++;
					struct _starpu_worker *_worker = _starpu_get_worker_struct(_workerid);
					_starpu_driver_start_job(_worker, j, &_worker->perf_arch, new_rank, profiling);
				}
			}
		}
		_STARPU_TRACE_TASK_COLOR(j);
		_STARPU_TRACE_START_CODELET_BODY(j, j->nimpl, &sched_ctx->perf_arch, workerid);
	}
	else
	{
		_STARPU_TRACE_TASK_COLOR(j);
		_STARPU_TRACE_START_CODELET_BODY(j, j->nimpl, perf_arch, workerid);
	}
	_STARPU_TASK_BREAK_ON(task, exec);
}

void _starpu_driver_end_job(struct _starpu_worker *worker, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch STARPU_ATTRIBUTE_UNUSED, int rank, int profiling)
{
	struct starpu_task *task = j->task;
	struct starpu_codelet *cl = task->cl;
	int starpu_top=_starpu_top_status_get();
	int workerid = worker->workerid;
	unsigned calibrate_model = 0;

	// Find out if the worker is the master of a parallel context
	struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(worker, j);
	if(!sched_ctx)
		sched_ctx = _starpu_get_sched_ctx_struct(j->task->sched_ctx);

	if (!sched_ctx->sched_policy)
	{
		_starpu_perfmodel_create_comb_if_needed(&(sched_ctx->perf_arch));
		_STARPU_TRACE_END_CODELET_BODY(j, j->nimpl, &(sched_ctx->perf_arch), workerid);
	}
	else
	{
		_starpu_perfmodel_create_comb_if_needed(perf_arch);
		_STARPU_TRACE_END_CODELET_BODY(j, j->nimpl, perf_arch, workerid);
	}

	if (cl && cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	if (rank == 0)
	{
		struct starpu_profiling_task_info *profiling_info = task->profiling_info;
		if ((profiling && profiling_info) || calibrate_model || starpu_top)
		{
			_starpu_clock_gettime(&worker->cl_end);
			_starpu_worker_register_executing_end(workerid);
		}
		STARPU_AYU_POSTRUNTASK(j->job_id);
	}

	if (starpu_top)
		_starpu_top_task_ended(task,workerid,&worker->cl_end);

	_starpu_set_worker_status(worker, STATUS_UNKNOWN);

	if(!sched_ctx->sched_policy && !sched_ctx->awake_workers &&
	   sched_ctx->main_master == worker->workerid)
	{
		struct starpu_worker_collection *workers = sched_ctx->workers;
		struct starpu_sched_ctx_iterator it;
		int new_rank = 0;

		if (workers->init_iterator)
			workers->init_iterator(workers, &it);
		while (workers->has_next(workers, &it))
		{
			int _workerid = workers->get_next(workers, &it);
			if (_workerid != workerid)
			{
				new_rank++;
				struct _starpu_worker *_worker = _starpu_get_worker_struct(_workerid);
				_starpu_driver_end_job(_worker, j, &_worker->perf_arch, new_rank, profiling);
			}
		}
	}
}

void _starpu_driver_update_job_feedback(struct _starpu_job *j, struct _starpu_worker *worker,
					struct starpu_perfmodel_arch* perf_arch,
					int profiling)
{
	struct starpu_profiling_task_info *profiling_info = j->task->profiling_info;
	struct timespec measured_ts;
	int workerid = worker->workerid;
	struct starpu_codelet *cl = j->task->cl;
	int calibrate_model = 0;
	int updated = 0;

	_starpu_perfmodel_create_comb_if_needed(perf_arch);

#ifndef STARPU_SIMGRID
	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;
#endif

	if ((profiling && profiling_info) || calibrate_model)
	{
		double measured;

		starpu_timespec_sub(&worker->cl_end, &worker->cl_start, &measured_ts);
		measured = starpu_timing_timespec_to_us(&measured_ts);

		if (profiling && profiling_info)
		{
			memcpy(&profiling_info->start_time, &worker->cl_start, sizeof(struct timespec));
			memcpy(&profiling_info->end_time, &worker->cl_end, sizeof(struct timespec));

			profiling_info->workerid = workerid;

			_starpu_worker_update_profiling_info_executing(workerid, &measured_ts, 1,
								       profiling_info->used_cycles,
								       profiling_info->stall_cycles,
								       profiling_info->energy_consumed);
			updated =  1;
		}

		if (calibrate_model)
		{
#ifdef STARPU_OPENMP
			double time_consumed = measured;
			unsigned do_update_time_model;
			if (j->continuation)
			{
				/* The job is only paused, thus we accumulate
				 * its timing, but we don't update its
				 * perfmodel now. */
				starpu_timespec_accumulate(&j->cumulated_ts, &measured_ts);
				do_update_time_model = 0;
			}
			else
			{
				if (j->discontinuous)
				{
					/* The job was paused at least once but is now
					 * really completing. We need to take into
					 * account its past execution time in its
					 * perfmodel. */
					starpu_timespec_accumulate(&measured_ts, &j->cumulated_ts);
					time_consumed = starpu_timing_timespec_to_us(&measured_ts);
				}
				do_update_time_model = 1;
			}
#else
			const unsigned do_update_time_model = 1;
			const double time_consumed = measured;
#endif
			if (do_update_time_model)
			{
				_starpu_update_perfmodel_history(j, j->task->cl->model, perf_arch, worker->devid, time_consumed, j->nimpl);
			}
		}
	}

	if (!updated)
		_starpu_worker_update_profiling_info_executing(workerid, NULL, 1, 0, 0, 0);

	if (profiling_info && profiling_info->energy_consumed && cl->energy_model && cl->energy_model->benchmarking)
	{
#ifdef STARPU_OPENMP
		double energy_consumed = profiling_info->energy_consumed;
		unsigned do_update_energy_model;
		if (j->continuation)
		{
			j->cumulated_energy_consumed += energy_consumed;
			do_update_energy_model = 0;
		}
		else
		{
			if (j->discontinuous)
			{
				energy_consumed += j->cumulated_energy_consumed;
			}
			do_update_energy_model = 1;
		}
#else
		const double energy_consumed = profiling_info->energy_consumed;
		const unsigned do_update_energy_model = 1;
#endif

		if (do_update_energy_model)
		{
			_starpu_update_perfmodel_history(j, j->task->cl->energy_model, perf_arch, worker->devid, energy_consumed, j->nimpl);
		}
	}
}

static void _starpu_worker_set_status_scheduling(int workerid)
{
	if (_starpu_worker_get_status(workerid) != STATUS_SLEEPING
		&& _starpu_worker_get_status(workerid) != STATUS_WAKING_UP
		&& _starpu_worker_get_status(workerid) != STATUS_SCHEDULING)
	{
		_STARPU_TRACE_WORKER_SCHEDULING_START;
		_starpu_worker_set_status(workerid, STATUS_SCHEDULING);
	}
}

static void _starpu_worker_set_status_scheduling_done(int workerid)
{
	if (_starpu_worker_get_status(workerid) == STATUS_SCHEDULING)
	{
		_STARPU_TRACE_WORKER_SCHEDULING_END;
		_starpu_worker_set_status(workerid, STATUS_UNKNOWN);
	}
}

static void _starpu_worker_set_status_sleeping(int workerid)
{
	if ( _starpu_worker_get_status(workerid) == STATUS_WAKING_UP)
		_starpu_worker_set_status(workerid, STATUS_SLEEPING);
	else if (_starpu_worker_get_status(workerid) != STATUS_SLEEPING)
	{
		_STARPU_TRACE_WORKER_SLEEP_START;
		_starpu_worker_restart_sleeping(workerid);
		_starpu_worker_set_status(workerid, STATUS_SLEEPING);
	}

}

static void _starpu_worker_set_status_wakeup(int workerid)
{
	if (_starpu_worker_get_status(workerid) == STATUS_SLEEPING || _starpu_worker_get_status(workerid) == STATUS_WAKING_UP)
	{
		_STARPU_TRACE_WORKER_SLEEP_END;
		_starpu_worker_stop_sleeping(workerid);
		_starpu_worker_set_status(workerid, STATUS_UNKNOWN);
	}
}


static void _starpu_exponential_backoff(struct _starpu_worker *worker)
{
	int delay = worker->spinning_backoff;

	if (worker->spinning_backoff < BACKOFF_MAX)
		worker->spinning_backoff<<=1;

	while(delay--)
		STARPU_UYIELD();
}



/* Workers may block when there is no work to do at all. */
struct starpu_task *_starpu_get_worker_task(struct _starpu_worker *worker, int workerid, unsigned memnode STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
	struct starpu_task *task;
	unsigned needed = 1;
	unsigned executing STARPU_ATTRIBUTE_UNUSED = 0;

	_starpu_worker_set_status_scheduling(workerid);
	while(needed)
	{
		struct _starpu_sched_ctx *sched_ctx = NULL;
		struct _starpu_sched_ctx_list *l = NULL;
		for (l = worker->sched_ctx_list; l; l = l->next)
		{
			sched_ctx = _starpu_get_sched_ctx_struct(l->sched_ctx);
			if(sched_ctx && sched_ctx->id > 0 && sched_ctx->id < STARPU_NMAX_SCHED_CTXS)
			{
				STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->parallel_sect_mutex[workerid]);
				if(!sched_ctx->sched_policy && sched_ctx->awake_workers) 
					worker->slave = sched_ctx->main_master != workerid;

				if(sched_ctx->parallel_sect[workerid])
				{
					/* don't let the worker sleep with the sched_mutex taken */
					/* we need it until here bc of the list of ctxs of the workers
					   that can change in another thread */
					STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
					needed = 0;
					_starpu_sched_ctx_signal_worker_blocked(sched_ctx->id, workerid);
					sched_ctx->busy[workerid] = 1;
					STARPU_PTHREAD_COND_WAIT(&sched_ctx->parallel_sect_cond[workerid], &sched_ctx->parallel_sect_mutex[workerid]);
					sched_ctx->busy[workerid] = 0;
					STARPU_PTHREAD_COND_SIGNAL(&sched_ctx->parallel_sect_cond_busy[workerid]);
					_starpu_sched_ctx_signal_worker_woke_up(sched_ctx->id, workerid);
					sched_ctx->parallel_sect[workerid] = 0;
					STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
				}
				STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->parallel_sect_mutex[workerid]);
			}
			if(!needed)
				break;
		}
		/* don't worry if the value is not correct (no lock) it will do it next time */
		if(worker->tmp_sched_ctx != -1)
		{
			sched_ctx = _starpu_get_sched_ctx_struct(worker->tmp_sched_ctx);
			STARPU_PTHREAD_MUTEX_LOCK(&sched_ctx->parallel_sect_mutex[workerid]);
			if(sched_ctx->parallel_sect[workerid])
			{
//				needed = 0;
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
				_starpu_sched_ctx_signal_worker_blocked(sched_ctx->id, workerid);
				sched_ctx->busy[workerid] = 1;
				STARPU_PTHREAD_COND_WAIT(&sched_ctx->parallel_sect_cond[workerid], &sched_ctx->parallel_sect_mutex[workerid]);
				sched_ctx->busy[workerid] = 0;
				STARPU_PTHREAD_COND_SIGNAL(&sched_ctx->parallel_sect_cond_busy[workerid]);
				_starpu_sched_ctx_signal_worker_woke_up(sched_ctx->id, workerid);
				sched_ctx->parallel_sect[workerid] = 0;
				STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
			}
			STARPU_PTHREAD_MUTEX_UNLOCK(&sched_ctx->parallel_sect_mutex[workerid]);
		}

		needed = !needed;
	}

	if ((worker->pipeline_length == 0 && worker->current_task)
		|| (worker->pipeline_length != 0 && worker->ntasks))
		/* This worker is executing something */
		executing = 1;

	if (worker->pipeline_length && (worker->ntasks == worker->pipeline_length || worker->pipeline_stuck))
		task = NULL;
	else
		task = _starpu_pop_task(worker);

	if (task == NULL && !executing)
	{
		/* Didn't get a task to run and none are running, go to sleep */

		/* Note: we need to keep the sched condition mutex all along the path
		 * from popping a task from the scheduler to blocking. Otherwise the
		 * driver may go block just after the scheduler got a new task to be
		 * executed, and thus hanging. */

		_starpu_worker_set_status_sleeping(workerid);

		if (_starpu_worker_can_block(memnode, worker)
#ifndef STARPU_SIMGRID
				&& !_starpu_sched_ctx_last_worker_awake(worker)
#endif
				)
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
		else
		{
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
			if (_starpu_machine_is_running())
			{
				_starpu_exponential_backoff(worker);
#ifdef STARPU_SIMGRID
				static int warned;
				if (!warned)
				{
					warned = 1;
					_STARPU_DISP("Has to make simgrid spin for CPU idle time.  You can try to pass --enable-blocking-drivers to ./configure to avoid this\n");
				}
				MSG_process_sleep(0.000010);
#endif
			}
		}

		return NULL;
	}

	if (task)
	{
		_starpu_worker_set_status_scheduling_done(workerid);
		_starpu_worker_set_status_wakeup(workerid);
	}
	else
	{
		_starpu_worker_set_status_sleeping(workerid);
	}
	worker->spinning_backoff = BACKOFF_MIN;

	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);


	STARPU_AYU_PRERUNTASK(_starpu_get_job_associated_to_task(task)->job_id, workerid);

	return task;
}


int _starpu_get_multi_worker_task(struct _starpu_worker *workers, struct starpu_task ** tasks, int nworkers, unsigned memnode STARPU_ATTRIBUTE_UNUSED)
{
	int i, count = 0;
	struct _starpu_job * j;
	int is_parallel_task;
	struct _starpu_combined_worker *combined_worker;
	int executing STARPU_ATTRIBUTE_UNUSED = 0;
	/*for each worker*/
#ifndef STARPU_NON_BLOCKING_DRIVERS
	/* This assumes only 1 worker */
	STARPU_ASSERT_MSG(nworkers == 1, "Multiple workers is not yet possible in blocking drivers mode\n");
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&workers[0].sched_mutex);
#endif
	for (i = 0; i < nworkers; i++)
	{
		if ((workers[i].pipeline_length == 0 && workers[i].current_task)
			|| (workers[i].pipeline_length != 0 && workers[i].ntasks))
			/* At least this worker is executing something */
			executing = 1;
		/*if the worker is already executing a task then */
		if((workers[i].pipeline_length == 0 && workers[i].current_task)
			|| (workers[i].pipeline_length != 0 &&
				(workers[i].ntasks == workers[i].pipeline_length
				 || workers[i].pipeline_stuck)))
		{
			tasks[i] = NULL;
		}
		/*else try to pop a task*/
		else
		{
#ifdef STARPU_NON_BLOCKING_DRIVERS
			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&workers[i].sched_mutex);
#endif
			_starpu_worker_set_status_scheduling(workers[i].workerid);
			_starpu_set_local_worker_key(&workers[i]);
			tasks[i] = _starpu_pop_task(&workers[i]);
			if(tasks[i] != NULL)
			{
				_starpu_worker_set_status_scheduling_done(workers[i].workerid);
				_starpu_worker_set_status_wakeup(workers[i].workerid);
#ifdef STARPU_NON_BLOCKING_DRIVERS
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&workers[i].sched_mutex);
#endif

				count ++;
				j = _starpu_get_job_associated_to_task(tasks[i]);
				is_parallel_task = (j->task_size > 1);
				if (workers[i].pipeline_length)
					workers[i].current_tasks[(workers[i].first_task + workers[i].ntasks)%STARPU_MAX_PIPELINE] = tasks[i];
				else
					workers[i].current_task = j->task;
				workers[i].ntasks++;
				/* Get the rank in case it is a parallel task */
				if (is_parallel_task)
				{

					STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
					workers[i].current_rank = j->active_task_alias_count++;
					STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

					if(j->combined_workerid != -1)
					{
						combined_worker = _starpu_get_combined_worker_struct(j->combined_workerid);
						workers[i].combined_workerid = j->combined_workerid;
						workers[i].worker_size = combined_worker->worker_size;
					}
				}
				else
				{
					workers[i].combined_workerid = workers[i].workerid;
					workers[i].worker_size = 1;
					workers[i].current_rank = 0;
				}
				STARPU_AYU_PRERUNTASK(_starpu_get_job_associated_to_task(tasks[i])->job_id, workers[i].workerid);
			}
			else
			{
				_starpu_worker_set_status_sleeping(workers[i].workerid);
#ifdef STARPU_NON_BLOCKING_DRIVERS
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&workers[i].sched_mutex);
#endif
			}
		}
	}

#ifndef STARPU_NON_BLOCKING_DRIVERS
	/* Block the assumed-to-be-only worker */
	struct _starpu_worker *worker = &workers[0];
	unsigned workerid = workers[0].workerid;

	if (!count && !executing)
	{
		/* Didn't get a task to run and none are running, go to sleep */

		/* Note: we need to keep the sched condition mutex all along the path
		 * from popping a task from the scheduler to blocking. Otherwise the
		 * driver may go block just after the scheduler got a new task to be
		 * executed, and thus hanging. */
		_starpu_worker_set_status_sleeping(workerid);

		if (_starpu_worker_can_block(memnode, worker)
#ifndef STARPU_SIMGRID
				&& !_starpu_sched_ctx_last_worker_awake(worker)
#endif
				)
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
		}
		else
		{
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
			if (_starpu_machine_is_running())
			{
				_starpu_exponential_backoff(worker);
#ifdef STARPU_SIMGRID
				static int warned;
				if (!warned)
				{
					warned = 1;
					_STARPU_DISP("Has to make simgrid spin for CPU idle time.  You can try to pass --enable-blocking-drivers to ./configure to avoid this\n");
				}
				MSG_process_sleep(0.000010);
#endif
			}
		}
		return 0;
	}

	_starpu_worker_set_status_wakeup(workerid);
	worker->spinning_backoff = BACKOFF_MIN;

	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
#endif /* !STARPU_NON_BLOCKING_DRIVERS */

	return count;
}
