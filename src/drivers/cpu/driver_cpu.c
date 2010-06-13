/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
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
#include "driver_cpu.h"
#include <core/policies/sched_policy.h>

static int execute_job_on_cpu(starpu_job_t j, struct starpu_worker_s *cpu_args)
{
	int ret;
	struct timespec codelet_start, codelet_end;
	struct timespec codelet_start_comm, codelet_end_comm;
	int64_t start_time;
	int64_t end_time;

	unsigned calibrate_model = 0;
	int workerid = cpu_args->workerid;
	struct starpu_task *task = j->task;
	struct starpu_codelet_t *cl = task->cl;

	STARPU_ASSERT(cl);
	STARPU_ASSERT(cl->cpu_func);

	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	if (calibrate_model || STARPU_BENCHMARK_COMM)
		starpu_clock_gettime(&codelet_start_comm);

	ret = _starpu_fetch_task_input(task, 0);

	if (calibrate_model || STARPU_BENCHMARK_COMM)
		starpu_clock_gettime(&codelet_end_comm);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return -EAGAIN;
	}

	STARPU_TRACE_START_CODELET_BODY(j);

	if (calibrate_model || STARPU_BENCHMARK_COMM)
		starpu_clock_gettime(&codelet_start);

	int profiling_status = starpu_profiling_status_get();

	if (profiling_status)
		start_time = (int64_t)_starpu_timing_now();

	cpu_args->status = STATUS_EXECUTING;
	task->status = STARPU_TASK_RUNNING;	

	cl_func func = cl->cpu_func;
	func(task->interface, task->cl_arg);

	cl->per_worker_stats[workerid]++;
	
	if (calibrate_model || STARPU_BENCHMARK_COMM)
		starpu_clock_gettime(&codelet_end);

	if (profiling_status)
		end_time = (int64_t)_starpu_timing_now();

	struct starpu_task_profiling_info *profiling_info;
	profiling_info = task->profiling_info;

	if (profiling_info)
	{
		profiling_info->start_time = start_time;
		profiling_info->end_time = end_time;
		profiling_info->workerid = workerid;
	}

	if (profiling_status)
		_starpu_worker_update_profiling_info(workerid, end_time - start_time, 0, 1);

	STARPU_TRACE_END_CODELET_BODY(j);
	cpu_args->status = STATUS_UNKNOWN;

	_starpu_push_task_output(task, 0);

	if (calibrate_model || STARPU_BENCHMARK_COMM)
	{
		double measured = _starpu_timing_timespec_delay_us(&codelet_start, &codelet_end);
		double measured_comm = _starpu_timing_timespec_delay_us(&codelet_start_comm, &codelet_end_comm);

//		fprintf(stderr, "%d\t%d\n", (int)j->penality, (int)measured_comm);
		cpu_args->jobq->total_computation_time += measured;
		cpu_args->jobq->total_communication_time += measured_comm;

		double error;
		error = fabs(STARPU_MAX(measured, 0.0) - STARPU_MAX(j->predicted, 0.0)); 
//		fprintf(stderr, "Error -> %le, predicted -> %le measured ->%le\n", error, j->predicted, measured);
		cpu_args->jobq->total_computation_time_error += error;

		if (calibrate_model)
			_starpu_update_perfmodel_history(j, cpu_args->arch, cpu_args->devid, measured);
	}

	(void)STARPU_ATOMIC_ADD(&cpu_args->jobq->total_job_performed, 1);

	return 0;
}

void *_starpu_cpu_worker(void *arg)
{
	struct starpu_worker_s *cpu_arg = arg;
	struct starpu_jobq_s *jobq = cpu_arg->jobq;
	unsigned memnode = cpu_arg->memory_node;
	int workerid = cpu_arg->workerid;
	int devid = cpu_arg->devid;

#ifdef STARPU_USE_FXT
	_starpu_fxt_register_thread(cpu_arg->bindid);
#endif
	STARPU_TRACE_WORKER_INIT_START(STARPU_FUT_CPU_KEY, devid, memnode);

	_starpu_bind_thread_on_cpu(cpu_arg->config, cpu_arg->bindid);

#ifdef STARPU_VERBOSE
        fprintf(stderr, "cpu worker %d is ready on logical cpu %d\n", devid, cpu_arg->bindid);
#endif

	_starpu_set_local_memory_node_key(&memnode);

	_starpu_set_local_queue(jobq);

	_starpu_set_local_worker_key(cpu_arg);

	snprintf(cpu_arg->name, 32, "CPU %d", devid);

	cpu_arg->status = STATUS_UNKNOWN;

	STARPU_TRACE_WORKER_INIT_END

        /* tell the main thread that we are ready */
	PTHREAD_MUTEX_LOCK(&cpu_arg->mutex);
	cpu_arg->worker_is_initialized = 1;
	PTHREAD_COND_SIGNAL(&cpu_arg->ready_cond);
	PTHREAD_MUTEX_UNLOCK(&cpu_arg->mutex);

        starpu_job_t j;
	int res;

	struct starpu_sched_policy_s *policy = _starpu_get_sched_policy();
	struct starpu_jobq_s *queue = policy->starpu_get_local_queue(policy);

	while (_starpu_machine_is_running())
	{
		STARPU_TRACE_START_PROGRESS(memnode);
		_starpu_datawizard_progress(memnode, 1);
		STARPU_TRACE_END_PROGRESS(memnode);

		_starpu_execute_registered_progression_hooks();

		_starpu_jobq_lock(queue);

		/* perhaps there is some local task to be executed first */
		j = _starpu_pop_local_task(cpu_arg);

		/* otherwise ask a task to the scheduler */
		if (!j)
			j = _starpu_pop_task();
		
                if (j == NULL) 
		{
			if (_starpu_worker_can_block(memnode))
				_starpu_block_worker(workerid, &queue->activity_cond, &queue->activity_mutex);

			_starpu_jobq_unlock(queue);

			continue;
		};
	
		_starpu_jobq_unlock(queue);
		
		/* can a cpu perform that task ? */
		if (!STARPU_CPU_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			_starpu_push_task(j, 0);
			continue;
		}

		_starpu_set_current_task(j->task);

                res = execute_job_on_cpu(j, cpu_arg);

		_starpu_set_current_task(NULL);

		if (res) {
			switch (res) {
				case -EAGAIN:
					_starpu_push_task(j, 0);
					continue;
				default: 
					assert(0);
			}
		}

		_starpu_handle_job_termination(j, 0);
        }

	STARPU_TRACE_WORKER_DEINIT_START

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

#ifdef STARPU_DATA_STATS
	fprintf(stderr, "CPU #%d computation %le comm %le (%lf \%%)\n", devid, jobq->total_computation_time, jobq->total_communication_time,  jobq->total_communication_time*100.0/jobq->total_computation_time);
#endif

#ifdef STARPU_VERBOSE
	double ratio = 0;
	if (jobq->total_job_performed != 0)
	{
		ratio = jobq->total_computation_time_error/jobq->total_computation_time;
	}

	_starpu_print_to_logfile("MODEL ERROR: CPU %d ERROR %lf EXEC %lf RATIO %lf NTASKS %d\n", devid, jobq->total_computation_time_error, jobq->total_computation_time, ratio, jobq->total_job_performed);
#endif

	STARPU_TRACE_WORKER_DEINIT_END(STARPU_FUT_CPU_KEY);

	pthread_exit(NULL);
}
