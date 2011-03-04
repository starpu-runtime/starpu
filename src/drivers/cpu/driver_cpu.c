/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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
#include <drivers/driver_common/driver_common.h>
#include <common/utils.h>
#include <core/debug.h>
#include "driver_cpu.h"
#include <core/sched_policy.h>

static int execute_job_on_cpu(starpu_job_t j, struct starpu_worker_s *cpu_args, int is_parallel_task, int rank, enum starpu_perf_archtype perf_arch)
{
	int ret;
	struct timespec codelet_start, codelet_end;

	unsigned calibrate_model = 0;
	int workerid = cpu_args->workerid;
	struct starpu_task *task = j->task;
	struct starpu_codelet_t *cl = task->cl;

	STARPU_ASSERT(cl);
	STARPU_ASSERT(cl->cpu_func);

	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	if (rank == 0)
	{
		ret = _starpu_fetch_task_input(task, 0);
		if (ret != 0)
		{
			/* there was not enough memory so the codelet cannot be executed right now ... */
			/* push the codelet back and try another one ... */
			return -EAGAIN;
		}
	}

	if (is_parallel_task)
		PTHREAD_BARRIER_WAIT(&j->before_work_barrier);

	STARPU_TRACE_START_CODELET_BODY(j);

	struct starpu_task_profiling_info *profiling_info;
	int profiling = starpu_profiling_status_get();

	if (rank == 0)
	{
		profiling_info = task->profiling_info;
	
		if ((profiling && profiling_info) || calibrate_model)
		{
			starpu_clock_gettime(&codelet_start);
			_starpu_worker_register_executing_start_date(workerid, &codelet_start);
		}

	}
	
	cpu_args->status = STATUS_EXECUTING;
	task->status = STARPU_TASK_RUNNING;	
	
	/* In case this is a Fork-join parallel task, the worker does not
	 * execute the kernel at all. */
	if ((rank == 0) || (cl->type != STARPU_FORKJOIN))
	{
		cl_func func = cl->cpu_func;
		func(task->interfaces, task->cl_arg);
	}
	
	if (is_parallel_task)
		PTHREAD_BARRIER_WAIT(&j->after_work_barrier);

	STARPU_TRACE_END_CODELET_BODY(j);

	cpu_args->status = STATUS_UNKNOWN;

	if (rank == 0)
	{
		cl->per_worker_stats[workerid]++;
		
		if ((profiling && profiling_info) || calibrate_model)
			starpu_clock_gettime(&codelet_end);

		_starpu_push_task_output(task, 0);

		_starpu_driver_update_job_feedback(j, cpu_args, profiling_info,
				perf_arch, &codelet_start, &codelet_end);
	}

	return 0;
}

void *_starpu_cpu_worker(void *arg)
{
	struct starpu_worker_s *cpu_arg = arg;
	unsigned memnode = cpu_arg->memory_node;
	int workerid = cpu_arg->workerid;
	int devid = cpu_arg->devid;

#ifdef STARPU_USE_FXT
	_starpu_fxt_register_thread(cpu_arg->bindid);
#endif
	STARPU_TRACE_WORKER_INIT_START(STARPU_FUT_CPU_KEY, devid, memnode);

	_starpu_bind_thread_on_cpu(cpu_arg->config, cpu_arg->bindid);

        _STARPU_DEBUG("cpu worker %d is ready on logical cpu %d\n", devid, cpu_arg->bindid);

	_starpu_set_local_memory_node_key(&memnode);

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
	struct starpu_task *task;

	int res;

	while (_starpu_machine_is_running())
	{
		STARPU_TRACE_START_PROGRESS(memnode);
		_starpu_datawizard_progress(memnode, 1);
		STARPU_TRACE_END_PROGRESS(memnode);

		PTHREAD_MUTEX_LOCK(cpu_arg->sched_mutex);

		task = _starpu_pop_task(cpu_arg);
	
                if (!task) 
		{
			if (_starpu_worker_can_block(memnode))
				_starpu_block_worker(workerid, cpu_arg->sched_cond, cpu_arg->sched_mutex);

			PTHREAD_MUTEX_UNLOCK(cpu_arg->sched_mutex);

			continue;
		};

		PTHREAD_MUTEX_UNLOCK(cpu_arg->sched_mutex);	

		STARPU_ASSERT(task);
		j = _starpu_get_job_associated_to_task(task);
	
		/* can a cpu perform that task ? */
		if (!STARPU_CPU_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			_starpu_push_task(j, 0);
			continue;
		}

		int rank = 0;
		int is_parallel_task = (j->task_size > 1);

		enum starpu_perf_archtype perf_arch; 
	
		/* Get the rank in case it is a parallel task */
		if (is_parallel_task)
		{
			/* We can release the fake task */
			STARPU_ASSERT(task != j->task);
			free(task);

			PTHREAD_MUTEX_LOCK(&j->sync_mutex);
			rank = j->active_task_alias_count++;
			PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

			struct starpu_combined_worker_s *combined_worker;
			combined_worker = _starpu_get_combined_worker_struct(j->combined_workerid);

			cpu_arg->combined_workerid = j->combined_workerid;
			cpu_arg->worker_size = combined_worker->worker_size;
			cpu_arg->current_rank = rank;
			perf_arch = combined_worker->perf_arch;
		}
		else {
			cpu_arg->combined_workerid = cpu_arg->workerid;
			cpu_arg->worker_size = 1;
			cpu_arg->current_rank = 0;
			perf_arch = cpu_arg->perf_arch;
		}

		_starpu_set_current_task(j->task);

                res = execute_job_on_cpu(j, cpu_arg, is_parallel_task, rank, perf_arch);

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

		if (rank == 0)
			_starpu_handle_job_termination(j, 0);
        }

	STARPU_TRACE_WORKER_DEINIT_START

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	STARPU_TRACE_WORKER_DEINIT_END(STARPU_FUT_CPU_KEY);

	pthread_exit(NULL);
}
