/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2020,2021  Federal University of Rio Grande do Sul (UFRGS)
 * Copyright (C) 2022-2023  Camille Coti
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
#include <drivers/mp_common/source_common.h>
#include <drivers/disk/driver_disk.h>
#include <drivers/max/driver_max_fpga.h>
#include <core/sched_policy.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <datawizard/datawizard.h>
#include <core/simgrid.h>
#include <core/task.h>
#include <core/disk.h>
#include <common/knobs.h>
#include <profiling/callbacks.h>

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

static unsigned already_busy_cpus;

static struct _starpu_driver_info driver_info =
{
	.name_upper = "CPU",
	.name_var = "CPU",
	.name_lower = "cpu",
	.memory_kind = STARPU_CPU_RAM,
	.alpha = 0.5f,
	.wait_for_worker_initialization = 1,
#ifdef STARPU_USE_CPU
	.driver_ops = &_starpu_driver_cpu_ops,
	.run_worker = _starpu_cpu_worker,
#endif
	.init_worker_binding = _starpu_cpu_init_worker_binding,
	.init_worker_memory = _starpu_cpu_init_worker_memory,
};

static struct _starpu_memory_driver_info memory_driver_info =
{
	.name_upper = "NUMA",
	.worker_archtype = STARPU_CPU_WORKER,
	.ops = &_starpu_driver_cpu_node_ops,
};

/* Early library initialization, before anything else, just initialize data */
void _starpu_cpu_preinit(void)
{
	_starpu_driver_info_register(STARPU_CPU_WORKER, &driver_info);
	_starpu_memory_driver_info_register(STARPU_CPU_RAM, &memory_driver_info);
	already_busy_cpus = 0;
}

void _starpu_cpu_busy_cpu(unsigned num)
{
	already_busy_cpus += num;
}

#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
/* Determine which devices we will use */
void _starpu_init_cpu_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config)
{
	int ncpu = config->conf.ncpus;

	if (ncpu != 0)
	{
		STARPU_ASSERT_MSG(ncpu >= -1, "ncpus can not be negative and different from -1 (is is %d)", ncpu);

		int nhyperthreads = topology->nhwpus / topology->nhwworker[STARPU_CPU_WORKER][0];
		long avail_cpus = (long) (topology->nusedpus / nhyperthreads) - (long) already_busy_cpus;
		if (avail_cpus < 0)
			avail_cpus = 0;
		int nth_per_core = starpu_getenv_number_default("STARPU_NTHREADS_PER_CORE", 1);
		avail_cpus *= nth_per_core;

		_starpu_topology_check_ndevices(&ncpu, avail_cpus, 1, STARPU_MAXCPUS, config->conf.reserve_ncpus, "ncpus", "CPU cores", "maxcpus");
	}

	topology->ndevices[STARPU_CPU_WORKER] = 1;
	unsigned homogeneous = starpu_getenv_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_CPU", 1);

	_starpu_topology_configure_workers(topology, config,
					   STARPU_CPU_WORKER,
					   0, 0, homogeneous, 1,
					   ncpu, 1, NULL, NULL);
}
#endif

/* Bind the driver on a CPU core */
void _starpu_cpu_init_worker_binding(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	/* Dedicate a cpu core to that worker */
	workerarg->bindid = _starpu_get_next_bindid(config, STARPU_THREAD_ACTIVE, NULL, 0);;
}

/* Set up memory and buses */
void _starpu_cpu_init_worker_memory(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg)
{
	unsigned memory_node = -1;
	int numa_logical_id = _starpu_get_logical_numa_node_worker(workerarg->workerid);
	int numa_starpu_id =  starpu_memory_nodes_numa_hwloclogid_to_id(numa_logical_id);
	if (numa_starpu_id < 0 || numa_starpu_id >= STARPU_MAXNUMANODES)
		numa_starpu_id = STARPU_MAIN_RAM;

#if defined(STARPU_HAVE_HWLOC) && !defined(STARPU_SIMGRID)
	hwloc_obj_t pu_obj = hwloc_get_obj_by_type(config->topology.hwtopology, HWLOC_OBJ_PU, workerarg->bindid);
	struct _starpu_hwloc_userdata *userdata = pu_obj->userdata;
	userdata->pu_worker = workerarg;
#endif

	workerarg->numa_memory_node = memory_node = numa_starpu_id;

	_starpu_memory_node_add_nworkers(memory_node);

	_starpu_worker_drives_memory_node(workerarg, numa_starpu_id);

	workerarg->memory_node = memory_node;
}

#ifdef STARPU_USE_CPU
/* This is run from the driver thread to initialize the driver CUDA context */
int _starpu_cpu_driver_init(struct _starpu_worker *cpu_worker)
{
	int devid = cpu_worker->devid;

#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init, devid, cpu_worker->workerid, starpu_prof_tool_driver_cpu, -1, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init(&pi, NULL, NULL);

	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init_start, devid, cpu_worker->workerid, starpu_prof_tool_driver_cpu, -1, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start(&pi, NULL, NULL);
#endif

	_starpu_driver_start(cpu_worker, STARPU_CPU_WORKER, 1);
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

#ifdef STARPU_PROF_TOOL
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init_end, devid, cpu_worker->workerid, starpu_prof_tool_driver_cpu, -1, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_end(&pi, NULL, NULL);
#endif
	return 0;
}

int _starpu_cpu_driver_deinit(struct _starpu_worker *cpu_worker)
{
	_STARPU_TRACE_WORKER_DEINIT_START;

	unsigned memnode = cpu_worker->memory_node;
	_starpu_datawizard_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	cpu_worker->worker_is_initialized = 0;
	_STARPU_TRACE_WORKER_DEINIT_END(STARPU_CPU_WORKER);

#ifdef STARPU_PROF_TOOL
 	int workerid = cpu_worker->workerid;
	struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_deinit, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit(&pi, NULL, NULL);
#endif

	return 0;
}
#endif /* STARPU_USE_CPU */

uintptr_t _starpu_cpu_malloc_on_device(int dst_dev, size_t size, int flags)
{
	uintptr_t addr = 0;
	unsigned dst_node = starpu_memory_devid_find_node(dst_dev, STARPU_CPU_RAM);
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

void _starpu_cpu_free_on_device(int dst_dev, uintptr_t addr, size_t size, int flags)
{
	unsigned dst_node = starpu_memory_devid_find_node(dst_dev, STARPU_CPU_RAM);
	_starpu_free_flags_on_node(dst_node, (void*)addr, size,
#if defined(STARPU_USE_CUDA) && !defined(STARPU_HAVE_CUDA_MEMCPY_PEER) && !defined(STARPU_SIMGRID)
				   flags & ~STARPU_MALLOC_PINNED
#else
				   flags
#endif
				   );
}

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
	{
		STARPU_ASSERT_MSG(copy_methods->any_to_any, "the interface '%s' does define neither ram_to_ram nor any_to_any copy method", handle->ops->name);
		copy_methods->any_to_any(src_interface, src_node, dst_interface, dst_node, req ? &req->async_channel : NULL);
	}
	return ret;
}

int _starpu_cpu_copy_data(uintptr_t src, size_t src_offset, int src_dev, uintptr_t dst, size_t dst_offset, int dst_dev, size_t size, struct _starpu_async_channel *async_channel)
{
	(void) async_channel;
	(void) src_dev;
	(void) dst_dev;

	memcpy((void *) (dst + dst_offset), (void *) (src + src_offset), size);
	return 0;
}

int _starpu_cpu_is_direct_access_supported(unsigned node, unsigned handling_node)
{
	(void) node;
	(void) handling_node;
	return 1;
}

uintptr_t _starpu_cpu_map(uintptr_t src, size_t src_offset, unsigned src_node, unsigned dst_node, size_t size, int *ret)
{
	(void) src_node;
	(void) dst_node;
	(void) size;

	*ret = 0;
	return src + src_offset;
}

int _starpu_cpu_unmap(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, unsigned dst_node, size_t size)
{
	(void) src;
	(void) src_offset;
	(void) src_node;
	(void) dst;
	(void) dst_node;
	(void) size;

	return 0;
}

int _starpu_cpu_update_map(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size)
{
	(void) src;
	(void) src_offset;
	(void) src_node;
	(void) dst;
	(void) dst_offset;
	(void) dst_node;
	(void) size;

	/* Memory mappings are cache-coherent */
	return 0;
}

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
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
	int devid = cpu_args->devid;
#endif

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

	_starpu_cl_func_t func = _starpu_task_get_cpu_nth_implementation(cl, j->nimpl);

	/* In case this is a Fork-join parallel task, the worker does not
	 * execute the kernel at all. */
	if ((rank == 0) || (cl->type != STARPU_FORKJOIN))
	{
		if (is_parallel_task && cl->type == STARPU_FORKJOIN)
			/* bind to parallel worker */
			_starpu_bind_thread_on_cpus(_starpu_get_combined_worker_struct(j->combined_workerid));
		STARPU_ASSERT_MSG(func, "when STARPU_CPU is defined in 'where', cpu_func or cpu_funcs has to be defined");
		if (_starpu_get_disable_kernels() <= 0)
		{
			_STARPU_TRACE_START_EXECUTING(j);
#ifdef STARPU_PROF_TOOL
			pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_cpu_exec, devid, worker_task->workerid, starpu_prof_tool_driver_cpu, -1, (void*)func);
			pi.model_name = _starpu_job_get_model_name( j );
			pi.task_name = _starpu_job_get_task_name( j );
			starpu_prof_tool_callbacks.starpu_prof_tool_event_start_cpu_exec(&pi, NULL, NULL);
#endif
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
			{
				struct _starpu_sched_ctx *sched_ctx = _starpu_sched_ctx_get_sched_ctx_for_worker_and_job(cpu_args, j);
				_starpu_simgrid_submit_job(cpu_args->workerid, sched_ctx->id, j, perf_arch, NAN, NAN, NULL);
			}
#else
#ifdef STARPU_PAPI
			if (rank == 0)
				_starpu_profiling_papi_task_start_counters(task);
#endif
			func(_STARPU_TASK_GET_INTERFACES(task), task->cl_arg);
#ifdef STARPU_PAPI
			if (rank == 0)
				_starpu_profiling_papi_task_stop_counters(task);
#endif
#endif
#ifdef STARPU_PROF_TOOL
			pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_cpu_exec, devid, worker_task->workerid, starpu_prof_tool_driver_cpu, -1, (void*)func);
			pi.model_name = _starpu_job_get_model_name( j );
			pi.task_name = _starpu_job_get_task_name( j );
			starpu_prof_tool_callbacks.starpu_prof_tool_event_end_cpu_exec(&pi, NULL, NULL);
#endif
			_STARPU_TRACE_END_EXECUTING(j);
		}
		if (is_parallel_task && cl->type == STARPU_FORKJOIN)
			/* rebind to single CPU */
			_starpu_bind_thread_on_cpu(cpu_args->bindid, cpu_args->workerid, NULL);
	}
	else
	{
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_cpu_exec, devid, worker_task->workerid, starpu_prof_tool_driver_cpu, -1, (void*)func);
		pi.model_name = _starpu_job_get_model_name( j );
		pi.task_name = _starpu_job_get_task_name( j );
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_cpu_exec(&pi, NULL, NULL);
#endif
		_STARPU_TRACE_START_EXECUTING(j);
	}

	if (is_parallel_task)
	{
		_STARPU_TRACE_START_PARALLEL_SYNC(j);
		STARPU_PTHREAD_BARRIER_WAIT(&j->after_work_barrier);
		_STARPU_TRACE_END_PARALLEL_SYNC(j);
		if (rank != 0)
		{
#ifdef STARPU_PROF_TOOL
			pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_cpu_exec, devid, worker_task->workerid, starpu_prof_tool_driver_cpu, -1, (void*)func);
			pi.model_name = _starpu_job_get_model_name( j );
			pi.task_name = _starpu_job_get_task_name( j );
			starpu_prof_tool_callbacks.starpu_prof_tool_event_end_cpu_exec(&pi, NULL, NULL);
#endif
			_STARPU_TRACE_END_EXECUTING(j);
		}
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
	j->workerid = cpu_worker->workerid;

#ifdef STARPU_BUBBLE_VERBOSE
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	unsigned long long timestamp = 1000000000ULL*tp.tv_sec + tp.tv_nsec;
	_STARPU_DEBUG("{%llu} [%s(%p)]\n", timestamp, starpu_task_get_name(task), task);
#endif
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

/* One iteration of the main driver loop */
int _starpu_cpu_driver_run_once(struct _starpu_worker *cpu_worker)
{
	unsigned memnode = cpu_worker->memory_node;
	int workerid = cpu_worker->workerid;
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif
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
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, cpu_worker->nb_buffers_totransfer, cpu_worker->nb_buffers_transferred);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
		j = _starpu_get_job_associated_to_task(pending_task);

		_starpu_fetch_task_input_tail(pending_task, j, cpu_worker);
		/* Reset it */
		cpu_worker->task_transferring = NULL;

		ret = _starpu_cpu_driver_execute_task(cpu_worker, pending_task, j);
		_STARPU_TRACE_START_PROGRESS(memnode);
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_start_transfer, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, cpu_worker->nb_buffers_totransfer, cpu_worker->nb_buffers_transferred);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
		return ret;
	}

	res = __starpu_datawizard_progress(_STARPU_DATAWIZARD_DO_ALLOC, 1);

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
	if (!_STARPU_MAY_PERFORM(j, CPU))
	{
		/* put it and the end of the queue ... XXX */
		_starpu_push_task_to_workers(task);
		return 0;
	}

#ifdef STARPU_PROF_TOOL
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, NULL);
	pi.model_name = _starpu_job_get_model_name( j );
	pi.task_name = _starpu_job_get_task_name( j );
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
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
#ifdef STARPU_PROF_TOOL
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, cpu_worker->workerid, starpu_prof_tool_driver_cpu, memnode, NULL);
		pi.model_name = _starpu_job_get_model_name( j );
		pi.task_name = _starpu_job_get_task_name( j );
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
		_STARPU_TRACE_END_PROGRESS(memnode);
		return ret;
	}
#ifdef STARPU_PROF_TOOL
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, cpu_worker->workerid, starpu_prof_tool_driver_cpu, memnode, NULL);
	pi.model_name = _starpu_job_get_model_name( j );
	pi.task_name = _starpu_job_get_task_name( j );
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
	_STARPU_TRACE_END_PROGRESS(memnode);
	return 0;
}

void *_starpu_cpu_worker(void *arg)
{
	struct _starpu_worker *worker = arg;

	_starpu_cpu_driver_init(worker);
	_STARPU_TRACE_START_PROGRESS(worker->memory_node);
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, starpu_prof_tool_driver_cpu, worker->memory_node, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
#endif
	while (_starpu_machine_is_running())
	{
		_starpu_may_pause();
		_starpu_cpu_driver_run_once(worker);
	}
	_STARPU_TRACE_END_PROGRESS(worker->memory_node);
#ifdef STARPU_PROF_TOOL
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, worker->workerid, worker->workerid, starpu_prof_tool_driver_cpu, worker->memory_node, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
#endif
	_starpu_cpu_driver_deinit(worker);

	return NULL;
}

int _starpu_cpu_driver_run(struct _starpu_worker *worker)
{
	_starpu_cpu_worker(worker);

	return 0;
}

int _starpu_cpu_driver_set_devid(struct starpu_driver *driver, struct _starpu_worker *worker)
{
	driver->id.cpu_id = worker->devid;

	return 0;
}

int _starpu_cpu_driver_is_devid(struct starpu_driver *driver, struct _starpu_worker *worker)
{
	return driver->id.cpu_id == worker->devid;
}

struct _starpu_driver_ops _starpu_driver_cpu_ops =
{
	.init = _starpu_cpu_driver_init,
	.run = _starpu_cpu_driver_run,
	.run_once = _starpu_cpu_driver_run_once,
	.deinit = _starpu_cpu_driver_deinit,
	.set_devid = _starpu_cpu_driver_set_devid,
	.is_devid = _starpu_cpu_driver_is_devid,
};
#endif /* STARPU_USE_CPU */

struct _starpu_node_ops _starpu_driver_cpu_node_ops =
{
	.name = "cpu driver",

	.malloc_on_device = _starpu_cpu_malloc_on_device,
	.free_on_device = _starpu_cpu_free_on_device,

	.is_direct_access_supported = _starpu_cpu_is_direct_access_supported,

	.copy_interface_to[STARPU_CPU_RAM] = _starpu_cpu_copy_interface,

	.copy_data_to[STARPU_CPU_RAM] = _starpu_cpu_copy_data,

	.map[STARPU_CPU_RAM] = _starpu_cpu_map,
	.unmap[STARPU_CPU_RAM] = _starpu_cpu_unmap,
	.update_map[STARPU_CPU_RAM] = _starpu_cpu_update_map,
};
