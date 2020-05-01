/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2020       Federal University of Rio Grande do Sul (UFRGS)
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

#include <common/config.h>

#include <math.h>
#include <starpu.h>
#include <starpu_profiling.h>
#include <drivers/driver_common/driver_common.h>
#include <common/utils.h>
#include <core/debug.h>
#include <core/workers.h>
#include <core/drivers.h>
#include <core/idle_hook.h>
#include <drivers/cpu/driver_cpu.h>
#include <drivers/disk/driver_disk.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/mic/driver_mic_source.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/disk/driver_disk.h>
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <core/simgrid.h>
#include <core/task.h>
#include <core/disk.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif


#ifdef STARPU_USE_CPU
/* Actually launch the job on a cpu worker.
 * Handle binding CPUs on cores.
 * In the case of a combined worker WORKER_TASK != J->TASK */

static int execute_job_on_cpu(struct _starpu_job *j, struct starpu_task *worker_task, struct _starpu_worker *cpu_args, int rank, struct starpu_perfmodel_arch* perf_arch)
{
	int is_parallel_task = (j->task_size > 1);
	int profiling = starpu_profiling_status_get();
	struct starpu_task *task = j->task;
	struct starpu_codelet *cl = task->cl;

	STARPU_ASSERT(cl);

	if (is_parallel_task)
	{
		STARPU_PTHREAD_BARRIER_WAIT(&j->before_work_barrier);

		/* In the case of a combined worker, the scheduler needs to know
		 * when each actual worker begins the execution */
		_starpu_sched_pre_exec_hook(worker_task);
	}

	/* Give profiling variable */
	_starpu_driver_start_job(cpu_args, j, perf_arch, rank, profiling);

	/* In case this is a Fork-join parallel task, the worker does not
	 * execute the kernel at all. */
	if ((rank == 0) || (cl->type != STARPU_FORKJOIN))
	{
		_starpu_cl_func_t func = _starpu_task_get_cpu_nth_implementation(cl, j->nimpl);
		if (is_parallel_task && cl->type == STARPU_FORKJOIN)
			/* bind to parallel worker */
			_starpu_bind_thread_on_cpus(_starpu_get_combined_worker_struct(j->combined_workerid));
		STARPU_ASSERT_MSG(func, "when STARPU_CPU is defined in 'where', cpu_func or cpu_funcs has to be defined");
		if (_starpu_get_disable_kernels() <= 0)
		{
			_STARPU_TRACE_START_EXECUTING();
#ifdef STARPU_SIMGRID
			if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE)
				func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
			else if (cl->flags & STARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT)
			{
				_SIMGRID_TIMER_BEGIN(1);
				func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
				_SIMGRID_TIMER_END;
			}
			else
				_starpu_simgrid_submit_job(cpu_args->workerid, j, perf_arch, NAN, NULL);
#else
			func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
#endif
			_STARPU_TRACE_END_EXECUTING();
		}
		if (is_parallel_task && cl->type == STARPU_FORKJOIN)
			/* rebind to single CPU */
			_starpu_bind_thread_on_cpu(cpu_args->bindid, cpu_args->workerid, NULL);
	}
	else
	{
		_STARPU_TRACE_START_EXECUTING();
	}

	if (is_parallel_task)
	{
		STARPU_PTHREAD_BARRIER_WAIT(&j->after_work_barrier);
		if (rank != 0)
			_STARPU_TRACE_END_EXECUTING();
	}

	_starpu_driver_end_job(cpu_args, j, perf_arch, rank, profiling);

	if (is_parallel_task)
	{
#ifdef STARPU_SIMGRID
		if (rank == 0)
		{
			/* Wait for other threads to exit barrier_wait so we
			 * can safely drop the job structure */
			starpu_sleep(0.0000001);
			j->after_work_busy_barrier = 0;
		}
#else
		ANNOTATE_HAPPENS_BEFORE(&j->after_work_busy_barrier);
		(void) STARPU_ATOMIC_ADD(&j->after_work_busy_barrier, -1);
		if (rank == 0)
		{
			/* Wait with a busy barrier for other workers to have
			 * finished with the blocking barrier before we can
			 * safely drop the job structure */
			while (j->after_work_busy_barrier > 0)
			{
				STARPU_UYIELD();
				STARPU_SYNCHRONIZE();
			}
			ANNOTATE_HAPPENS_AFTER(&j->after_work_busy_barrier);
		}
#endif
	}

	if (rank == 0)
	{
		_starpu_driver_update_job_feedback(j, cpu_args, perf_arch, profiling);
#ifdef STARPU_OPENMP
		if (!j->continuation)
#endif
		{
			_starpu_push_task_output(j);
		}
	}

	return 0;
}

int _starpu_cpu_driver_init(struct _starpu_worker *cpu_worker)
{
	int devid = cpu_worker->devid;

	_starpu_driver_start(cpu_worker, _STARPU_FUT_CPU_KEY, 1);
	snprintf(cpu_worker->name, sizeof(cpu_worker->name), "CPU %d", devid);
	snprintf(cpu_worker->short_name, sizeof(cpu_worker->short_name), "CPU %d", devid);
	starpu_pthread_setname(cpu_worker->short_name);

	_STARPU_TRACE_WORKER_INIT_END(cpu_worker->workerid);

	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&cpu_worker->sched_mutex);
	cpu_worker->status = STATUS_UNKNOWN;
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&cpu_worker->sched_mutex);

	/* tell the main thread that we are ready */
	STARPU_PTHREAD_MUTEX_LOCK(&cpu_worker->mutex);
	cpu_worker->worker_is_initialized = 1;
	STARPU_PTHREAD_COND_SIGNAL(&cpu_worker->ready_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&cpu_worker->mutex);
	return 0;
}

static int _starpu_cpu_driver_execute_task(struct _starpu_worker *cpu_worker, struct starpu_task *task, struct _starpu_job *j)
{
	int res;

	int rank;
	int is_parallel_task = (j->task_size > 1);

	struct starpu_perfmodel_arch* perf_arch;

	rank = cpu_worker->current_rank;

	/* Get the rank in case it is a parallel task */
	if (is_parallel_task)
	{
		if(j->combined_workerid != -1)
		{
			struct _starpu_combined_worker *combined_worker;
			combined_worker = _starpu_get_combined_worker_struct(j->combined_workerid);

			cpu_worker->combined_workerid = j->combined_workerid;
			cpu_worker->worker_size = combined_worker->worker_size;
			perf_arch = &combined_worker->perf_arch;
		}
		else
		{
			struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(cpu_worker, j);
			STARPU_ASSERT_MSG(sched_ctx != NULL, "there should be a worker %d in the ctx of this job \n", cpu_worker->workerid);

			perf_arch = &sched_ctx->perf_arch;
		}
	}
	else
	{
		cpu_worker->combined_workerid = cpu_worker->workerid;
		cpu_worker->worker_size = 1;

		struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(cpu_worker, j);
		if (sched_ctx && !sched_ctx->sched_policy && !sched_ctx->awake_workers && sched_ctx->main_master == cpu_worker->workerid)
			perf_arch = &sched_ctx->perf_arch;
		else
			perf_arch = &cpu_worker->perf_arch;
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
			_starpu_push_task_to_workers(task);
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

int _starpu_cpu_driver_run_once(struct _starpu_worker *cpu_worker)
{
	unsigned memnode = cpu_worker->memory_node;
	int workerid = cpu_worker->workerid;

	int res;

	struct _starpu_job *j;
	struct starpu_task *task = NULL, *pending_task;

	int rank = 0;

#ifdef STARPU_SIMGRID
	starpu_pthread_wait_reset(&cpu_worker->wait);
#endif

	/* Test if async transfers are completed */
	pending_task = cpu_worker->task_transferring;
	if (pending_task != NULL && cpu_worker->nb_buffers_transferred == cpu_worker->nb_buffers_totransfer)
	{
		int ret;
		STARPU_RMB();
		_STARPU_TRACE_END_PROGRESS(memnode);
		j = _starpu_get_job_associated_to_task(pending_task);

		_starpu_fetch_task_input_tail(pending_task, j, cpu_worker);
		_starpu_set_worker_status(cpu_worker, STATUS_UNKNOWN);
		/* Reset it */
		cpu_worker->task_transferring = NULL;

		ret = _starpu_cpu_driver_execute_task(cpu_worker, pending_task, j);
		_STARPU_TRACE_START_PROGRESS(memnode);
		return ret;
	}

	res = __starpu_datawizard_progress(1, 1);

	if (!pending_task)
		task = _starpu_get_worker_task(cpu_worker, workerid, memnode);

#ifdef STARPU_SIMGRID
 #ifndef STARPU_OPENMP
	if (!res && !task)
		/* No progress, wait */
		starpu_pthread_wait_wait(&cpu_worker->wait);
 #else
  #if SIMGRID_VERSION >= 31800
	if (!res && !task)
	{
		/* No progress, wait (but at most 1s for OpenMP support) */
		/* TODO: ideally, make OpenMP wake worker when run_once should return */
		struct timespec abstime;
		_starpu_clock_gettime(&abstime);
		abstime.tv_sec++;
		starpu_pthread_wait_timedwait(&cpu_worker->wait, &abstime);
	}
  #else
	/* Previous simgrid versions don't really permit to use wait_timedwait in C */
	starpu_sleep(0.001);
  #endif
 #endif
#endif

	if (!task)
	{
		/* No task or task still pending transfers */
		_starpu_execute_registered_idle_hooks();
		return 0;
	}

	j = _starpu_get_job_associated_to_task(task);
	/* NOTE: j->task is != task for parallel tasks, which share the same
	 * job. */

	/* can a cpu perform that task ? */
	if (!_STARPU_CPU_MAY_PERFORM(j))
	{
		/* put it and the end of the queue ... XXX */
		_starpu_push_task_to_workers(task);
		return 0;
	}

	_STARPU_TRACE_END_PROGRESS(memnode);
	/* Get the rank in case it is a parallel task */
	if (j->task_size > 1)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&j->sync_mutex);
		rank = j->active_task_alias_count++;
		STARPU_PTHREAD_MUTEX_UNLOCK(&j->sync_mutex);
	}
	else
	{
		rank = 0;
	}
	cpu_worker->current_rank = rank;

#ifdef STARPU_OPENMP
	/* At this point, j->continuation as been cleared as the task is being
	 * woken up, thus we use j->discontinuous instead for the check */
	const unsigned continuation_wake_up = j->discontinuous;
#else
	const unsigned continuation_wake_up = 0;
#endif
	if (rank == 0 && !continuation_wake_up)
	{
		res = _starpu_fetch_task_input(task, j, 1);
		STARPU_ASSERT(res == 0);
	}
	else
	{
		int ret = _starpu_cpu_driver_execute_task(cpu_worker, task, j);
		_STARPU_TRACE_END_PROGRESS(memnode);
		return ret;
	}
	_STARPU_TRACE_END_PROGRESS(memnode);
	return 0;
}

int _starpu_cpu_driver_deinit(struct _starpu_worker *cpu_worker)
{
	_STARPU_TRACE_WORKER_DEINIT_START;

	unsigned memnode = cpu_worker->memory_node;
	_starpu_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	cpu_worker->worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(_STARPU_FUT_CPU_KEY);

	return 0;
}

void *_starpu_cpu_worker(void *arg)
{
	struct _starpu_worker *worker = arg;

	_starpu_cpu_driver_init(worker);
	_STARPU_TRACE_START_PROGRESS(worker->memory_node);
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_cpu_driver_run_once(worker);
	}
	_STARPU_TRACE_END_PROGRESS(worker->memory_node);
	_starpu_cpu_driver_deinit(worker);

	return NULL;
}

int _starpu_cpu_driver_run(struct _starpu_worker *worker)
{
	worker->set = NULL;
	worker->worker_is_initialized = 0;
	_starpu_cpu_worker(worker);

	return 0;
}

struct _starpu_driver_ops _starpu_driver_cpu_ops =
{
	.init = _starpu_cpu_driver_init,
	.run = _starpu_cpu_driver_run,
	.run_once = _starpu_cpu_driver_run_once,
	.deinit = _starpu_cpu_driver_deinit
};
#endif /* STARPU_USE_CPU */

int _starpu_cpu_copy_interface(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_CPU_RAM);

	int ret = 0;
	const struct starpu_data_copy_methods *copy_methods = handle->ops->copy_methods;
	if (copy_methods->ram_to_ram)
		copy_methods->ram_to_ram(src_interface, src_node, dst_interface, dst_node);
	else
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req ? &req->async_channel : NULL);
	return ret;
}

int _starpu_cpu_copy_data(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel)
{
	int src_kind = starpu_node_get_kind(src_node);
	int dst_kind = starpu_node_get_kind(dst_node);
	STARPU_ASSERT(src_kind == STARPU_CPU_RAM && dst_kind == STARPU_CPU_RAM);

	(void) async_channel;

	memcpy((void *) (dst + dst_offset), (void *) (src + src_offset), size);
	return 0;
}

int _starpu_cpu_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	(void) node;
	(void) handling_node;
	return 1;
}

uintptr_t _starpu_cpu_malloc_on_node(unsigned dst_node, size_t size, int flags)
{
	uintptr_t addr = 0;
	_starpu_malloc_flags_on_node(dst_node, (void**) &addr, size,
#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
				     /* without memcpy_peer, we can not
				      * allocated pinned memory, since it
				      * requires waiting for a task, and we
				      * may be called with a spinlock held
				      */
				     flags & ~STARPU_MALLOC_PINNED
#else
				     flags
#endif
				     );
	return addr;
}

void _starpu_cpu_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags)
{
	_starpu_free_flags_on_node(dst_node, (void*)addr, size,
#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
				   flags & ~STARPU_MALLOC_PINNED
#else
				   flags
#endif
				   );
}

struct _starpu_node_ops _starpu_driver_cpu_node_ops =
{
	.copy_interface_to[STARPU_UNUSED] = NULL,
	.copy_interface_to[STARPU_CPU_RAM] = _starpu_cpu_copy_interface,
#ifdef STARPU_USE_CUDA
	.copy_interface_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_interface_from_cpu_to_cuda,
#else
	.copy_interface_to[STARPU_CUDA_RAM] = NULL,
#endif
#ifdef STARPU_USE_OPENCL
	.copy_interface_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_interface_from_cpu_to_opencl,
#else
	.copy_interface_to[STARPU_OPENCL_RAM] = NULL,
#endif
	.copy_interface_to[STARPU_DISK_RAM] = _starpu_disk_copy_interface_from_cpu_to_disk,
#ifdef STARPU_USE_MIC
	.copy_interface_to[STARPU_MIC_RAM] = _starpu_mic_copy_interface_from_cpu_to_mic,
#else
	.copy_interface_to[STARPU_MIC_RAM] = NULL,
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	.copy_interface_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_interface_from_cpu_to_mpi,
#else
	.copy_interface_to[STARPU_MPI_MS_RAM] = NULL,
#endif

	.copy_data_to[STARPU_UNUSED] = NULL,
	.copy_data_to[STARPU_CPU_RAM] = _starpu_cpu_copy_data,
#ifdef STARPU_USE_CUDA
	.copy_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy_data_from_cpu_to_cuda,
#else
	.copy_data_to[STARPU_CUDA_RAM] = NULL,
#endif
#ifdef STARPU_USE_OPENCL
	.copy_data_to[STARPU_OPENCL_RAM] = _starpu_opencl_copy_data_from_cpu_to_opencl,
#else
	.copy_data_to[STARPU_OPENCL_RAM] = NULL,
#endif
	.copy_data_to[STARPU_DISK_RAM] = _starpu_disk_copy_data_from_cpu_to_disk,
#ifdef STARPU_USE_MIC
	.copy_data_to[STARPU_MIC_RAM] = _starpu_mic_copy_data_from_cpu_to_mic,
#else
	.copy_data_to[STARPU_MIC_RAM] = NULL,
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	.copy_data_to[STARPU_MPI_MS_RAM] = _starpu_mpi_copy_data_from_cpu_to_mpi,
#else
	.copy_data_to[STARPU_MPI_MS_RAM] = NULL,
#endif

	.copy2d_data_to[STARPU_UNUSED] = NULL,
	.copy2d_data_to[STARPU_CPU_RAM] = NULL,
#ifdef STARPU_USE_CUDA
	.copy2d_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy2d_data_from_cpu_to_cuda,
#else
	.copy2d_data_to[STARPU_CUDA_RAM] = NULL,
#endif
	.copy2d_data_to[STARPU_OPENCL_RAM] = NULL,
	.copy2d_data_to[STARPU_DISK_RAM] = NULL,
	.copy2d_data_to[STARPU_MIC_RAM] = NULL,
	.copy2d_data_to[STARPU_MPI_MS_RAM] = NULL,

	.copy3d_data_to[STARPU_UNUSED] = NULL,
	.copy3d_data_to[STARPU_CPU_RAM] = NULL,
#if 0
#ifdef STARPU_USE_CUDA
	.copy3d_data_to[STARPU_CUDA_RAM] = _starpu_cuda_copy3d_data_from_cpu_to_cuda,
#else
	.copy3d_data_to[STARPU_CUDA_RAM] = NULL,
#endif
#else
	.copy3d_data_to[STARPU_CUDA_RAM] = NULL,
#endif
	.copy3d_data_to[STARPU_OPENCL_RAM] = NULL,
	.copy3d_data_to[STARPU_DISK_RAM] = NULL,
	.copy3d_data_to[STARPU_MIC_RAM] = NULL,
	.copy3d_data_to[STARPU_MPI_MS_RAM] = NULL,

	.wait_request_completion = NULL,
	.test_request_completion = NULL,
	.is_direct_access_supported = _starpu_cpu_is_direct_access_supported,
	.malloc_on_node = _starpu_cpu_malloc_on_node,
	.free_on_node = _starpu_cpu_free_on_node,
	.name = "cpu driver"
};
