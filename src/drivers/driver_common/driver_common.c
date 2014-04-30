/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2014  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2014  Inria
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

void _starpu_driver_start_job(struct _starpu_worker *args, struct _starpu_job *j, struct timespec *codelet_start, int rank, int profiling)
{
	struct starpu_task *task = j->task;
	struct starpu_codelet *cl = task->cl;
	struct starpu_profiling_task_info *profiling_info;
	int starpu_top=_starpu_top_status_get();
	int workerid = args->workerid;
	unsigned calibrate_model = 0;

	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	/* If the job is executed on a combined worker there is no need for the
	 * scheduler to process it : it doesn't contain any valuable data
	 * as it's not linked to an actual worker */
	if (j->task_size == 1)
		_starpu_sched_pre_exec_hook(task);

	args->status = STATUS_EXECUTING;
	task->status = STARPU_TASK_RUNNING;

	if (rank == 0)
	{
#ifdef HAVE_AYUDAME_H
		if (AYU_event) AYU_event(AYU_RUNTASK, j->job_id, NULL);
#endif
		cl->per_worker_stats[workerid]++;

		profiling_info = task->profiling_info;

		if ((profiling && profiling_info) || calibrate_model || starpu_top)
		{
			_starpu_clock_gettime(codelet_start);
			_starpu_worker_register_executing_start_date(workerid, codelet_start);
		}
	}

	if (starpu_top)
		_starpu_top_task_started(task,workerid,codelet_start);

	_STARPU_TRACE_START_CODELET_BODY(j);
}

void _starpu_driver_end_job(struct _starpu_worker *args, struct _starpu_job *j, struct starpu_perfmodel_arch* perf_arch STARPU_ATTRIBUTE_UNUSED, struct timespec *codelet_end, int rank, int profiling)
{
	struct starpu_task *task = j->task;
	struct starpu_codelet *cl = task->cl;
	struct starpu_profiling_task_info *profiling_info = task->profiling_info;
	int starpu_top=_starpu_top_status_get();
	int workerid = args->workerid;
	unsigned calibrate_model = 0;

	_STARPU_TRACE_END_CODELET_BODY(j, j->nimpl, perf_arch);

	if (cl && cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	if (rank == 0)
	{
		if ((profiling && profiling_info) || calibrate_model || starpu_top)
			_starpu_clock_gettime(codelet_end);
#ifdef HAVE_AYUDAME_H
		if (AYU_event) AYU_event(AYU_POSTRUNTASK, j->job_id, NULL);
#endif
	}

	if (starpu_top)
		_starpu_top_task_ended(task,workerid,codelet_end);

	args->status = STATUS_UNKNOWN;
}
void _starpu_driver_update_job_feedback(struct _starpu_job *j, struct _starpu_worker *worker_args,
					struct starpu_perfmodel_arch* perf_arch,
					struct timespec *codelet_start, struct timespec *codelet_end, int profiling)
{
	struct starpu_profiling_task_info *profiling_info = j->task->profiling_info;
	struct timespec measured_ts;
	double measured;
	int workerid = worker_args->workerid;
	struct starpu_codelet *cl = j->task->cl;
	int calibrate_model = 0;
	int updated = 0;

#ifndef STARPU_SIMGRID
	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;
#endif

	if ((profiling && profiling_info) || calibrate_model)
	{
		starpu_timespec_sub(codelet_end, codelet_start, &measured_ts);
		measured = starpu_timing_timespec_to_us(&measured_ts);

		if (profiling && profiling_info)
		{
			memcpy(&profiling_info->start_time, codelet_start, sizeof(struct timespec));
			memcpy(&profiling_info->end_time, codelet_end, sizeof(struct timespec));

			profiling_info->workerid = workerid;

			_starpu_worker_update_profiling_info_executing(workerid, &measured_ts, 1,
								       profiling_info->used_cycles,
								       profiling_info->stall_cycles,
								       profiling_info->power_consumed);
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
				_starpu_update_perfmodel_history(j, j->task->cl->model, perf_arch, worker_args->devid, time_consumed, j->nimpl);
			}
		}
	}

	if (!updated)
		_starpu_worker_update_profiling_info_executing(workerid, NULL, 1, 0, 0, 0);

	if (profiling_info && profiling_info->power_consumed && cl->power_model && cl->power_model->benchmarking)
	{
#ifdef STARPU_OPENMP
		double power_consumed = profiling_info->power_consumed;
		unsigned do_update_power_model;
		if (j->continuation)
		{
			j->cumulated_power_consumed += power_consumed;
			do_update_power_model = 0;
		}
		else 
		{
			if (j->discontinuous)
			{
				power_consumed += j->cumulated_power_consumed;
			}
			do_update_power_model = 1;
		}
#else
		const double power_consumed = profiling_info->power_consumed;
		const unsigned do_update_power_model = 1;
#endif

		if (do_update_power_model)
		{
			_starpu_update_perfmodel_history(j, j->task->cl->power_model, perf_arch, worker_args->devid, power_consumed, j->nimpl);
		}
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


static void _starpu_exponential_backoff(struct _starpu_worker *args)
{
	int delay = args->spinning_backoff;
	
	if (args->spinning_backoff < BACKOFF_MAX)
		args->spinning_backoff<<=1; 
	
	while(delay--)
		STARPU_UYIELD();
}



/* Workers may block when there is no work to do at all. */
struct starpu_task *_starpu_get_worker_task(struct _starpu_worker *args, int workerid, unsigned memnode)
{
	struct starpu_task *task;

	STARPU_PTHREAD_MUTEX_LOCK(&args->parallel_sect_mutex);
	if(args->parallel_sect)
	{
		_starpu_sched_ctx_signal_worker_blocked(args->workerid);
		STARPU_PTHREAD_COND_WAIT(&args->parallel_sect_cond, &args->parallel_sect_mutex);
		starpu_sched_ctx_bind_current_thread_to_cpuid(args->bindid);
		_starpu_sched_ctx_signal_worker_woke_up(workerid);
		args->parallel_sect = 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&args->parallel_sect_mutex);

	STARPU_PTHREAD_MUTEX_LOCK(&args->sched_mutex);
	task = _starpu_pop_task(args);

	if (task == NULL)
	{
		/* Note: we need to keep the sched condition mutex all along the path
		 * from popping a task from the scheduler to blocking. Otherwise the
		 * driver may go block just after the scheduler got a new task to be
		 * executed, and thus hanging. */

		_starpu_worker_set_status_sleeping(workerid);

		if (_starpu_worker_can_block(memnode) && !_starpu_sched_ctx_last_worker_awake(args))
		{
			STARPU_PTHREAD_COND_WAIT(&args->sched_cond, &args->sched_mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&args->sched_mutex);
		}
		else
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&args->sched_mutex);			
			if (_starpu_machine_is_running())
			{
				_starpu_exponential_backoff(args);
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

	STARPU_PTHREAD_MUTEX_UNLOCK(&args->sched_mutex);

	_starpu_worker_set_status_wakeup(workerid);
	args->spinning_backoff = BACKOFF_MIN;


#ifdef HAVE_AYUDAME_H
	if (AYU_event)
	{
		intptr_t id = workerid;
		AYU_event(AYU_PRERUNTASK, _starpu_get_job_associated_to_task(task)->job_id, &id);
	}
#endif

	return task;
}


int _starpu_get_multi_worker_task(struct _starpu_worker *workers, struct starpu_task ** tasks, int nworkers)
{
	int i, count = 0;
	struct _starpu_job * j;
	int is_parallel_task;
	struct _starpu_combined_worker *combined_worker;
	/*for each worker*/
	for (i = 0; i < nworkers; i++)
	{
		/*if the worker is already executinf a task then */
		if(workers[i].current_task)
		{
			tasks[i] = NULL;
		}
		/*else try to pop a task*/
		else
		{
			STARPU_PTHREAD_MUTEX_LOCK(&workers[i].sched_mutex);
			_starpu_set_local_worker_key(&workers[i]);
			tasks[i] = _starpu_pop_task(&workers[i]);
			STARPU_PTHREAD_MUTEX_UNLOCK(&workers[i].sched_mutex);
			if(tasks[i] != NULL)
			{
				count ++;
				j = _starpu_get_job_associated_to_task(tasks[i]);
				is_parallel_task = (j->task_size > 1);
				workers[i].current_task = j->task;
				/* Get the rank in case it is a parallel task */
				if (is_parallel_task)
				{

					STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
					workers[i].current_rank = j->active_task_alias_count++;
					STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
					
					combined_worker = _starpu_get_combined_worker_struct(j->combined_workerid);
					workers[i].combined_workerid = j->combined_workerid;
					workers[i].worker_size = combined_worker->worker_size;
				}
				else
				{
					workers[i].combined_workerid = workers[i].workerid;
					workers[i].worker_size = 1;
					workers[i].current_rank = 0;
				}

				_starpu_worker_set_status_wakeup(workers[i].workerid);
			}
			else
			{
				_starpu_worker_set_status_sleeping(workers[i].workerid);
			}
		}
	}
	return count;
}

