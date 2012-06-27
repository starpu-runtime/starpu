/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2012  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#include <math.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <drivers/driver_common/driver_common.h>
#include <common/utils.h>
#include <core/debug.h>
#include "driver_cpu.h"
#include <core/sched_policy.h>

/* Actually launch the job on a cpu worker.
 * Handle binding CPUs on cores.
 * In the case of a combined worker WORKER_TASK != J->TASK */

static int execute_job_on_cpu(struct _starpu_job *j, struct starpu_task *worker_task, struct _starpu_worker *cpu_args, int rank, enum starpu_perf_archtype perf_arch)
{
	int ret;
	int is_parallel_task = (j->task_size > 1);
	int profiling = starpu_profiling_status_get();
	struct timespec codelet_start, codelet_end;

	struct starpu_task *task = j->task;
	struct starpu_codelet *cl = task->cl;

	STARPU_ASSERT(cl);

	if (rank == 0)
	{
		ret = _starpu_fetch_task_input(j, 0);
		if (ret != 0)
		{
			/* there was not enough memory so the codelet cannot be executed right now ... */
			/* push the codelet back and try another one ... */
			return -EAGAIN;
		}
	}

	if (is_parallel_task)
	{
		_STARPU_PTHREAD_BARRIER_WAIT(&j->before_work_barrier);

		/* In the case of a combined worker, the scheduler needs to know
		 * when each actual worker begins the execution */
		_starpu_sched_pre_exec_hook(worker_task);
	}

	/* Give profiling variable */
	_starpu_driver_start_job(cpu_args, j, &codelet_start, rank, profiling);

	/* In case this is a Fork-join parallel task, the worker does not
	 * execute the kernel at all. */
	if ((rank == 0) || (cl->type != STARPU_FORKJOIN))
	{
		_starpu_cl_func_t func = _starpu_task_get_cpu_nth_implementation(cl, j->nimpl);
		if (is_parallel_task && cl->type == STARPU_FORKJOIN)
			/* bind to parallel worker */
			_starpu_bind_thread_on_cpus(cpu_args->config, _starpu_get_combined_worker_struct(j->combined_workerid));
		STARPU_ASSERT(func);
		func(task->interfaces, task->cl_arg);
		if (is_parallel_task && cl->type == STARPU_FORKJOIN)
			/* rebind to single CPU */
			_starpu_bind_thread_on_cpu(cpu_args->config, cpu_args->bindid);
	}

	_starpu_driver_end_job(cpu_args, j, perf_arch, &codelet_end, rank, profiling);

	if (is_parallel_task)
		_STARPU_PTHREAD_BARRIER_WAIT(&j->after_work_barrier);

	if (rank == 0)
	{
		_starpu_driver_update_job_feedback(j, cpu_args,
				perf_arch, &codelet_start, &codelet_end, profiling);
		_starpu_push_task_output(j, 0);
	}

	return 0;
}

static struct _starpu_worker*
_starpu_get_worker_from_driver(struct starpu_driver *d)
{
	int workers[d->id.cpu_id + 1];
	int nworkers;
	nworkers = starpu_worker_get_ids_by_type(STARPU_CPU_WORKER, workers, d->id.cpu_id+1);
	if (nworkers >= 0 && (unsigned) nworkers < d->id.cpu_id)
		return NULL; // No device was found.
	
	return _starpu_get_worker_struct(workers[d->id.cpu_id]);
}

int _starpu_cpu_driver_init(struct starpu_driver *d)
{
	struct _starpu_worker *cpu_worker;
	cpu_worker = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(cpu_worker);

	unsigned memnode = cpu_worker->memory_node;
	int devid = cpu_worker->devid;

#ifdef STARPU_USE_FXT
	_starpu_fxt_register_thread(cpu_worker->bindid);
#endif
	_STARPU_TRACE_WORKER_INIT_START(_STARPU_FUT_CPU_KEY, devid, memnode);

	_starpu_bind_thread_on_cpu(cpu_worker->config, cpu_worker->bindid);

        _STARPU_DEBUG("cpu worker %d is ready on logical cpu %d\n", devid, cpu_worker->bindid);

	_starpu_set_local_memory_node_key(&cpu_worker->memory_node);

	_starpu_set_local_worker_key(cpu_worker);

	snprintf(cpu_worker->name, sizeof(cpu_worker->name), "CPU %d", devid);
	snprintf(cpu_worker->short_name, sizeof(cpu_worker->short_name), "CPU %d", devid);

	cpu_worker->status = STATUS_UNKNOWN;

	_STARPU_TRACE_WORKER_INIT_END

        /* tell the main thread that we are ready */
	_STARPU_PTHREAD_MUTEX_LOCK(&cpu_worker->mutex);
	cpu_worker->worker_is_initialized = 1;
	_STARPU_PTHREAD_COND_SIGNAL(&cpu_worker->ready_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&cpu_worker->mutex);
	return 0;
}

int _starpu_cpu_driver_run_once(struct starpu_driver *d)
{
	struct _starpu_worker *cpu_worker;
	cpu_worker = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(cpu_worker);

	unsigned memnode = cpu_worker->memory_node;
	int workerid = cpu_worker->workerid;

	_STARPU_TRACE_START_PROGRESS(memnode);
	_starpu_datawizard_progress(memnode, 1);
	_STARPU_TRACE_END_PROGRESS(memnode);

	/* Note: we need to keep the sched condition mutex all along the path
	 * from popping a task from the scheduler to blocking. Otherwise the
	 * driver may go block just after the scheduler got a new task to be
	 * executed, and thus hanging. */
	_STARPU_PTHREAD_MUTEX_LOCK(cpu_worker->sched_mutex);

        struct _starpu_job *j;
	struct starpu_task *task;
	int res;

	task = _starpu_pop_task(cpu_worker);

	if (!task)
	{
		if (_starpu_worker_can_block(memnode))
			_starpu_block_worker(workerid, cpu_worker->sched_cond, cpu_worker->sched_mutex);

		_STARPU_PTHREAD_MUTEX_UNLOCK(cpu_worker->sched_mutex);

		return 0;
	};

	_STARPU_PTHREAD_MUTEX_UNLOCK(cpu_worker->sched_mutex);

	STARPU_ASSERT(task);
	j = _starpu_get_job_associated_to_task(task);

	/* can a cpu perform that task ? */
	if (!_STARPU_CPU_MAY_PERFORM(j))
	{
		/* put it and the end of the queue ... XXX */
		_starpu_push_task(j);
		return 0;
	}

	int rank = 0;
	int is_parallel_task = (j->task_size > 1);

	enum starpu_perf_archtype perf_arch;

	/* Get the rank in case it is a parallel task */
	if (is_parallel_task)
	{
		_STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		rank = j->active_task_alias_count++;
		_STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);

		struct _starpu_combined_worker *combined_worker;
		combined_worker = _starpu_get_combined_worker_struct(j->combined_workerid);

		cpu_worker->combined_workerid = j->combined_workerid;
		cpu_worker->worker_size = combined_worker->worker_size;
		cpu_worker->current_rank = rank;
		perf_arch = combined_worker->perf_arch;
	}
	else
	{
		cpu_worker->combined_workerid = cpu_worker->workerid;
		cpu_worker->worker_size = 1;
		cpu_worker->current_rank = 0;
		perf_arch = cpu_worker->perf_arch;
	}

	_starpu_set_current_task(j->task);
	cpu_worker->current_task = j->task;

	res = execute_job_on_cpu(j, task, cpu_worker, rank, perf_arch);

	_starpu_set_current_task(NULL);
	cpu_worker->current_task = NULL;

	if (res)
	{
		switch (res)
		{
		case -EAGAIN:
			_starpu_push_task(j);
			return 0;
		default:
			STARPU_ABORT();
		}
	}

	/* In the case of combined workers, we need to inform the
	 * scheduler each worker's execution is over.
	 * Then we free the workers' task alias */
	if (is_parallel_task)
	{
		_starpu_sched_post_exec_hook(task);
		free(task);
	}

	if (rank == 0)
		_starpu_handle_job_termination(j);
	return 0;
}

int _starpu_cpu_driver_deinit(struct starpu_driver *d)
{
	_STARPU_TRACE_WORKER_DEINIT_START

	struct _starpu_worker *cpu_worker;
	cpu_worker = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(cpu_worker);

	unsigned memnode = cpu_worker->memory_node;
	_starpu_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_CPU_KEY);

	return 0;
}

void *
_starpu_cpu_worker(void *arg)
{
	struct _starpu_worker *args = arg;
	struct starpu_driver d =
	{
		.type      = STARPU_CPU_WORKER,
		.id.cpu_id = args->devid
	};

	_starpu_cpu_driver_init(&d);
	while (_starpu_machine_is_running())
		_starpu_cpu_driver_run_once(&d);
	_starpu_cpu_driver_deinit(&d);

	return NULL;
}

int _starpu_run_cpu(struct starpu_driver *d)
{
	STARPU_ASSERT(d && d->type == STARPU_CPU_WORKER);

	struct _starpu_worker *worker = _starpu_get_worker_from_driver(d);
	STARPU_ASSERT(worker);

	worker->set = NULL;
	worker->worker_is_initialized = 0;
	_starpu_cpu_worker(worker);

	return 0;
}
