/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016-2016  Uppsala University
 * Copyright (C) 2013-2013  Thibaut Lambert
 * Copyright (C) 2011-2011  Télécom Sud Paris
 * Copyright (C) 2021-2021  Federal University of Rio Grande do Sul (UFRGS)
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

#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#ifdef __linux__
#include <sys/utsname.h>
#endif
#include <common/config.h>
#include <common/utils.h>
#include <common/graph.h>
#include <core/progress_hook.h>
#include <core/idle_hook.h>
#include <core/workers.h>
#include <core/debug.h>
#include <core/disk.h>
#include <core/task.h>
#include <core/detect_combined_workers.h>
#include <datawizard/malloc.h>
#include <profiling/starpu_tracing.h>
#include <profiling/profiling.h>
#include <drivers/max/driver_max_fpga.h>
#include <profiling/bound.h>
#include <sched_policies/sched_component.h>
#include <datawizard/memory_nodes.h>
#include <common/knobs.h>
#include <drivers/mp_common/sink_common.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/mpi/driver_mpi_common.h>
#include <drivers/tcpip/driver_tcpip_common.h>

#include <drivers/cpu/driver_cpu.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/hip/driver_hip.h>
#include <drivers/opencl/driver_opencl.h>
#include <drivers/max/driver_max_fpga.h>
#include <drivers/mpi/driver_mpi_source.h>
#include <drivers/tcpip/driver_tcpip_source.h>
#include <drivers/disk/driver_disk.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

#ifdef STARPU_NOSV
#include <nosv.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#include <starpu_profiling_tool.h>

#if defined(_WIN32)
#ifdef __GNUC__
#define ffs(arg) __builtin_ffs(arg)
#else
#define ffs(arg) _bit_scan_forward(arg)
#endif
#endif


static int asynchronous_copy_disabled[STARPU_MAX_RAM+1];

/* global knobs */
static int __g_calibrate_knob;
static int __g_enable_catch_signal_knob;

/* per-worker knobs */
static int __w_bind_to_pu_knob;
static int __w_enable_worker_knob;

static struct starpu_perf_knob_group * __kg_starpu_global;
static struct starpu_perf_knob_group * __kg_starpu_worker__per_worker;

static void global_knobs__set(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value)
{
	/* context is not used for global knobs */
	STARPU_ASSERT(context == NULL);
	(void)context;

	if (knob->id == __g_calibrate_knob)
	{
		_starpu_set_calibrate_flag((unsigned)value->val_int32_t);
	}
	else if (knob->id == __g_enable_catch_signal_knob)
	{
		_starpu_set_catch_signals(!!value->val_int32_t);
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}
static void global_knobs__get(const struct starpu_perf_knob * const knob, void *context,       struct starpu_perf_knob_value * const value)
{
	/* context is not used for global knobs */
	STARPU_ASSERT(context == NULL);
	(void)context;

	if (knob->id == __g_calibrate_knob)
	{
		value->val_int32_t = (int32_t)_starpu_get_calibrate_flag();
	}
	else if (knob->id == __g_enable_catch_signal_knob)
	{
		value->val_int32_t = _starpu_get_catch_signals();
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}

static void worker_knobs__set(const struct starpu_perf_knob * const knob, void *context, const struct starpu_perf_knob_value * const value)
{
	const unsigned workerid = *(unsigned *)context;
	struct _starpu_worker * const worker = _starpu_get_worker_struct(workerid);
	if (knob->id == __w_bind_to_pu_knob)
	{
		STARPU_ASSERT(value->val_int32_t >= 0);
		worker->bindid_requested = value->val_int32_t;
	}
	else if (knob->id == __w_enable_worker_knob)
	{
		worker->enable_knob = !!value->val_int32_t;
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}
static void worker_knobs__get(const struct starpu_perf_knob * const knob, void *context,       struct starpu_perf_knob_value * const value)
{
	const unsigned workerid = *(unsigned *)context;
	struct _starpu_worker * const worker = _starpu_get_worker_struct(workerid);
	if (knob->id == __w_bind_to_pu_knob)
	{
		value->val_int32_t = worker->bindid;
	}
	else if (knob->id == __w_enable_worker_knob)
	{
		value->val_int32_t = worker->enable_knob;
	}
	else
	{
		STARPU_ASSERT(0);
		abort();
	}
}

void _starpu__workers_c__register_knobs(void)
{
	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_global;
		__kg_starpu_global = _starpu_perf_knob_group_register(scope, global_knobs__set, global_knobs__get);
		__STARPU_PERF_KNOB_REG("starpu.global", __kg_starpu_global, g_calibrate_knob, int32, "enable or disable performance models calibration (override STARPU_CALIBRATE env var)");
		__STARPU_PERF_KNOB_REG("starpu.global", __kg_starpu_global, g_enable_catch_signal_knob, int32, "enable or disable signal catching (override STARPU_CATCH_SIGNALS env var)");
	}

	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_per_worker;
		__kg_starpu_worker__per_worker = _starpu_perf_knob_group_register(scope, worker_knobs__set, worker_knobs__get);
		__STARPU_PERF_KNOB_REG("starpu.worker", __kg_starpu_worker__per_worker, w_bind_to_pu_knob, int32, "bind worker to PU (PU logical number, override StarPU binding env vars)");
		__STARPU_PERF_KNOB_REG("starpu.worker", __kg_starpu_worker__per_worker, w_enable_worker_knob, int32, "enable assigning task to that worker (1:Enabled | [0:Disabled])");
	}

#if 0
	{
		const enum starpu_perf_knob_scope scope = starpu_perf_knob_scope_per_scheduler;
		__kg_starpu_worker__per_scheduler = _starpu_perf_knob_group_register(scope, sched_knobs__set, sched_knobs__get);
	}
#endif
}

void _starpu__workers_c__unregister_knobs(void)
{
	_starpu_perf_knob_group_unregister(__kg_starpu_global);
	_starpu_perf_knob_group_unregister(__kg_starpu_worker__per_worker);
	__kg_starpu_global = NULL;
	__kg_starpu_worker__per_worker = NULL;
}

/* acquire/release semantic for concurrent initialization/de-initialization */
static starpu_pthread_mutex_t init_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static starpu_pthread_cond_t init_cond = STARPU_PTHREAD_COND_INITIALIZER;
static int init_count = 0;
static enum initialization initialized = UNINITIALIZED;

int _starpu_keys_initialized;
starpu_pthread_key_t _starpu_worker_key;
starpu_pthread_key_t _starpu_worker_set_key;

struct _starpu_machine_config _starpu_config;

static int check_entire_platform;

int _starpu_worker_parallel_blocks;

/* Pointers to argc and argv
 */
static int *my_argc = 0;
static char ***my_argv = NULL;

void _starpu__workers_c__register_kobs(void)
{
	/* TODO */
}

struct _starpu_driver_info starpu_driver_info[STARPU_NARCH];

void _starpu_driver_info_register(enum starpu_worker_archtype archtype, const struct _starpu_driver_info *info)
{
	STARPU_ASSERT(archtype < STARPU_NARCH);
	starpu_driver_info[archtype] = *info;
}

struct _starpu_memory_driver_info starpu_memory_driver_info[STARPU_MAX_RAM+1];

void _starpu_memory_driver_info_register(enum starpu_node_kind kind, const struct _starpu_memory_driver_info *info)
{
	starpu_memory_driver_info[kind] = *info;
}

/* Initialize value of static argc and argv, called when the process begins
 */
void _starpu_set_argc_argv(int *argc_param, char ***argv_param)
{
	my_argc = argc_param;
	my_argv = argv_param;
}

int *_starpu_get_argc()
{
	return my_argc;
}

char ***_starpu_get_argv()
{
	return my_argv;
}

int starpu_is_initialized(void)
{
	return initialized != UNINITIALIZED;
}

void starpu_wait_initialized(void)
{
	STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
	while (initialized != INITIALIZED)
		STARPU_PTHREAD_COND_WAIT(&init_cond, &init_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);
}

/* Makes sure that at least one of the workers of type <arch> can execute
 * <task>, for at least one of its implementations. */
static uint32_t _starpu_worker_exists_and_can_execute(struct starpu_task *task,
						      enum starpu_worker_archtype arch)
{
	_starpu_codelet_check_deprecated_fields(task->cl);

	/* make sure there is a worker on the machine able to execute the
	   task, independent of the sched_ctx, this latter may receive latter on
	   the necessary worker - the user or the hypervisor should take care this happens */

	if (check_entire_platform && !task->cl->can_execute)
	{
		if (!_starpu_get_machine_config()->topology.ndevices[arch])
			return 0;

		unsigned impl;
		for (impl = 0; impl < STARPU_MAXIMPLEMENTATIONS; impl++)
		{
			switch (arch)
			{
			case STARPU_CPU_WORKER:
				if (task->cl->cpu_funcs[impl] != NULL)
					return 1;
				break;
			case STARPU_CUDA_WORKER:
				if (task->cl->cuda_funcs[impl] != NULL)
					return 1;
				break;
			case STARPU_HIP_WORKER:
				if (task->cl->hip_funcs[impl] != NULL)
					return 1;
				break;
			case STARPU_OPENCL_WORKER:
				if (task->cl->opencl_funcs[impl] != NULL)
					return 1;
				break;
			case STARPU_MAX_FPGA_WORKER:
				if (task->cl->max_fpga_funcs[impl] != NULL)
					return 1;
				break;
			case STARPU_MPI_MS_WORKER:
			case STARPU_TCPIP_MS_WORKER:
				if (task->cl->cpu_funcs_name[impl] != NULL)
					return 1;
				break;
			default:
				STARPU_ABORT();
			}
		}
		return 0;
	}

	struct _starpu_sched_ctx *sched_ctx = _starpu_get_sched_ctx_struct(task->sched_ctx);
	struct starpu_worker_collection *workers = sched_ctx->workers;
	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		int i = workers->get_next(workers, &it);
		if (starpu_worker_get_type(i) != arch)
			continue;

		unsigned impl;
		for (impl = 0; impl < STARPU_MAXIMPLEMENTATIONS; impl++)
		{
			/* We could call task->cl->can_execute(i, task, impl)
			   here, it would definitely work. It is probably
			   cheaper to check whether it is necessary in order to
			   avoid a useless function call, though. */
			unsigned test_implementation = 0;
			switch (arch)
			{
			case STARPU_CPU_WORKER:
				if (task->cl->cpu_funcs[impl] != NULL)
					test_implementation = 1;
				break;
			case STARPU_CUDA_WORKER:
				if (task->cl->cuda_funcs[impl] != NULL)
					test_implementation = 1;
				break;
			case STARPU_HIP_WORKER:
				if (task->cl->hip_funcs[impl] != NULL)
					test_implementation = 1;
				break;
			case STARPU_OPENCL_WORKER:
				if (task->cl->opencl_funcs[impl] != NULL)
					test_implementation = 1;
				break;
			case STARPU_MAX_FPGA_WORKER:
				if (task->cl->max_fpga_funcs[impl] != NULL)
					test_implementation = 1;
				break;
			case STARPU_MPI_MS_WORKER:
			case STARPU_TCPIP_MS_WORKER:
				if (task->cl->cpu_funcs_name[impl] != NULL)
					test_implementation = 1;
				break;
			default:
				STARPU_ABORT();
			}

			if (!test_implementation)
				/* No implementation here, cannot execute */
				continue;

			if (task->cl->can_execute && !task->cl->can_execute(i, task, impl))
				/* The implementation cannot be executed here */
				continue;

			return 1;
		}
	}

	return 0;
}

/*
 * in case a task is submitted, we may check whether there exists a worker
 * that may execute the task or not
 */
uint32_t _starpu_worker_exists(struct starpu_task *task)
{
	_starpu_codelet_check_deprecated_fields(task->cl);
	if (task->where == STARPU_NOWHERE)
		return 1;

	/* if the task belongs to the init context we can
	   check out all the worker mask of the machine
	   if not we should iterate on the workers of the ctx
	   and verify if it exists a worker able to exec the task */
	if(task->sched_ctx == 0)
	{
		if (!(task->where & _starpu_config.worker_mask))
			return 0;

		if (!task->cl->can_execute)
			return 1;
	}

#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
	if ((task->where & STARPU_CPU) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_CPU_WORKER))
		return 1;
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if ((task->where & STARPU_CUDA) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_CUDA_WORKER))
		return 1;
#endif
#if defined(STARPU_USE_HIP)
	if ((task->where & STARPU_HIP) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_HIP_WORKER))
		return 1;
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if ((task->where & STARPU_OPENCL) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_OPENCL_WORKER))
		return 1;
#endif
#if defined(STARPU_USE_MAX_FPGA)
	if ((task->where & STARPU_MAX_FPGA) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_MAX_FPGA_WORKER))
		return 1;
#endif
#ifdef STARPU_USE_MPI_MASTER_SLAVE
	if ((task->where & STARPU_MPI_MS) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_MPI_MS_WORKER))
		return 1;
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	if ((task->where & STARPU_TCPIP_MS) &&
	    _starpu_worker_exists_and_can_execute(task, STARPU_TCPIP_MS_WORKER))
		return 1;
#endif

	return 0;
}

uint32_t _starpu_can_submit_ms_task(void)
{
	return (STARPU_MPI_MS & _starpu_config.worker_mask) || (STARPU_TCPIP_MS & _starpu_config.worker_mask) ;
}

uint32_t _starpu_can_submit_cuda_task(void)
{
	return STARPU_CUDA & _starpu_config.worker_mask;
}

uint32_t _starpu_can_submit_hip_task(void)
{
	return STARPU_HIP & _starpu_config.worker_mask;
}

uint32_t _starpu_can_submit_cpu_task(void)
{
	return STARPU_CPU & _starpu_config.worker_mask;
}

uint32_t _starpu_can_submit_opencl_task(void)
{
	return STARPU_OPENCL & _starpu_config.worker_mask;
}

static inline int _starpu_can_use_nth_implementation(enum starpu_worker_archtype arch, struct starpu_codelet *cl, unsigned nimpl)
{
	switch(arch)
	{
	case STARPU_ANY_WORKER:
	{
		int cpu_func_enabled=1, cuda_func_enabled=1, hip_func_enabled=1, opencl_func_enabled=1;

#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
		starpu_cpu_func_t cpu_func = _starpu_task_get_cpu_nth_implementation(cl, nimpl);
		cpu_func_enabled = cpu_func != NULL && starpu_cpu_worker_get_count();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
		starpu_cuda_func_t cuda_func = _starpu_task_get_cuda_nth_implementation(cl, nimpl);
		cuda_func_enabled = cuda_func != NULL && starpu_cuda_worker_get_count();
#endif
#if defined(STARPU_USE_HIP)
		starpu_hip_func_t hip_func = _starpu_task_get_hip_nth_implementation(cl, nimpl);
		hip_func_enabled = hip_func != NULL && starpu_hip_worker_get_count();
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
		starpu_opencl_func_t opencl_func = _starpu_task_get_opencl_nth_implementation(cl, nimpl);
		opencl_func_enabled = opencl_func != NULL && starpu_opencl_worker_get_count();
#endif

		return cpu_func_enabled && cuda_func_enabled && opencl_func_enabled && hip_func_enabled;
	}
	case STARPU_CPU_WORKER:
	{
		starpu_cpu_func_t func = _starpu_task_get_cpu_nth_implementation(cl, nimpl);
		return func != NULL;
	}
	case STARPU_CUDA_WORKER:
	{
		starpu_cuda_func_t func = _starpu_task_get_cuda_nth_implementation(cl, nimpl);
		return func != NULL;
	}
	case STARPU_HIP_WORKER:
	{
		starpu_hip_func_t func = _starpu_task_get_hip_nth_implementation(cl, nimpl);
		return func != NULL;
	}
	case STARPU_OPENCL_WORKER:
	{
		starpu_opencl_func_t func = _starpu_task_get_opencl_nth_implementation(cl, nimpl);
		return func != NULL;
	}
	case STARPU_MAX_FPGA_WORKER:
	{
		starpu_max_fpga_func_t func = _starpu_task_get_fpga_nth_implementation(cl, nimpl);
		return func != NULL;
	}
	case STARPU_MPI_MS_WORKER:
	case STARPU_TCPIP_MS_WORKER:
	{
		const char *func_name = _starpu_task_get_cpu_name_nth_implementation(cl, nimpl);
		return func_name != NULL;
	}
	default:
		STARPU_ASSERT_MSG(0, "Unknown arch type %d", arch);
	}
	return 0;
}

int _starpu_enforce_locality(unsigned workerid, struct starpu_task *task)
{
	if (!_starpu_config.conf.data_locality_enforce)
		return 1;

	unsigned i, requested_node = starpu_worker_get_memory_node(workerid);
	int owner = -1, shared=-1;
	for (i = 0; i < STARPU_TASK_GET_NBUFFERS(task); i++)
		if (STARPU_TASK_GET_MODE(task, i) & STARPU_LOCALITY)
		{
			starpu_data_handle_t handle = STARPU_TASK_GET_HANDLE(task, i);
			unsigned nnodes = starpu_memory_nodes_get_count();
			unsigned node;
			for (node = 0; node < nnodes; node++)
			{
				enum _starpu_cache_state state = handle->per_node[node].state;
				if (state == STARPU_OWNER)
					owner = node;
				if (state == STARPU_SHARED)
					shared = node;
			}
			if (owner != -1 && owner != (int)requested_node)
				return 0;
			if (shared != -1 && handle->per_node[requested_node].state != STARPU_SHARED)
				return 0;
		}
	return 1;
}

/* Test if this task can be processed on this worker, regardless of the implementation */
/* must be called with sched_mutex locked to protect state_blocked */
static inline int _starpu_can_execute_task_any_impl(unsigned workerid, struct starpu_task *task)
{
	if (!_starpu_enforce_locality(workerid, task))
		return 0;

	if (!_starpu_config.workers[workerid].enable_knob)
		return 0;

	if (task->workerids_len)
	{
		size_t div = sizeof(*task->workerids) * 8;
		if (workerid / div >= task->workerids_len || ! (task->workerids[workerid / div] & (1UL << workerid % div)))
			return 0;
	}

	/* if the worker is blocked in a parallel ctx don't submit tasks on it */
#ifdef STARPU_DEVEL
#warning FIXME: this is very expensive, while can_execute is supposed to be not very costly so schedulers can call it a lot
#endif
	if(starpu_worker_is_blocked_in_parallel(workerid))
		return 0;

	if (!(task->where & _starpu_config.workers[workerid].worker_mask))
		return 0;

	return 1;
}

/* must be called with sched_mutex locked to protect state_blocked_in_parallel */
int starpu_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	/* TODO: check that the task operand sizes will fit on that device */
	return _starpu_can_execute_task_any_impl(workerid, task) &&
		_starpu_can_use_nth_implementation(_starpu_config.workers[workerid].arch, task->cl, nimpl) &&
		(!task->cl->can_execute || task->cl->can_execute(workerid, task, nimpl));
}

/* must be called with sched_mutex locked to protect state_blocked_in_parallel */
int starpu_worker_can_execute_task_impl(unsigned workerid, struct starpu_task *task, unsigned *impl_mask)
{
	if (!_starpu_can_execute_task_any_impl(workerid, task))
		return 0;

	unsigned mask;
	int i;
	enum starpu_worker_archtype arch;
	struct starpu_codelet *cl;
	/* TODO: check that the task operand sizes will fit on that device */
	cl = task->cl;

	mask = 0;
	arch = _starpu_config.workers[workerid].arch;
	if (!task->cl->can_execute)
	{
		for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
			if (_starpu_can_use_nth_implementation(arch, cl, i))
			{
				mask |= 1U << i;
				if (!impl_mask)
					break;
			}
	}
	else
	{
		for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
			if (_starpu_can_use_nth_implementation(arch, cl, i)
			 && (!task->cl->can_execute || task->cl->can_execute(workerid, task, i)))
			{
				mask |= 1U << i;
				if (!impl_mask)
					break;
			}
	}
	if (impl_mask)
		*impl_mask = mask;
	return mask != 0;
}

/* must be called with sched_mutex locked to protect state_blocked */
int starpu_worker_can_execute_task_first_impl(unsigned workerid, struct starpu_task *task, unsigned *nimpl)
{
	if (!_starpu_can_execute_task_any_impl(workerid, task))
		return 0;
	int i;
	enum starpu_worker_archtype arch;
	struct starpu_codelet *cl;
	/* TODO: check that the task operand sizes will fit on that device */
	cl = task->cl;

	arch = _starpu_config.workers[workerid].arch;
	if (!task->cl->can_execute)
	{
		for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
			if (_starpu_can_use_nth_implementation(arch, cl, i))
			{
				if (nimpl)
					*nimpl = i;
				return 1;
			}
	}
	else
	{
		for (i = 0; i < STARPU_MAXIMPLEMENTATIONS; i++)
			if (_starpu_can_use_nth_implementation(arch, cl, i)
			 && (task->cl->can_execute(workerid, task, i)))
			{
				if (nimpl)
					*nimpl = i;
				return 1;
			}
	}
	return 0;
}



int starpu_combined_worker_can_execute_task(unsigned workerid, struct starpu_task *task, unsigned nimpl)
{
	/* TODO: check that the task operand sizes will fit on that device */

	struct starpu_codelet *cl = task->cl;
	unsigned nworkers = _starpu_config.topology.nworkers;


	/* Is this a parallel worker ? */
	if (workerid < nworkers)
	{
		if (!_starpu_can_execute_task_any_impl(workerid, task))
			return 0;

		return !!((task->where & _starpu_config.workers[workerid].worker_mask) &&
				_starpu_can_use_nth_implementation(_starpu_config.workers[workerid].arch, task->cl, nimpl) &&
				(!task->cl->can_execute || task->cl->can_execute(workerid, task, nimpl)));
	}
	else
	{
		if (!_starpu_enforce_locality(workerid, task))
			return 0;

		if (task->workerids_len)
		{
			size_t div = sizeof(*task->workerids) * 8;
			if (workerid / div >= task->workerids_len || ! (task->workerids[workerid / div] & (1UL << workerid % div)))
				return 0;
		}

		if (cl->type == STARPU_SPMD
#ifdef STARPU_HAVE_HWLOC
				|| cl->type == STARPU_FORKJOIN
#else
#ifdef __GLIBC__
				|| cl->type == STARPU_FORKJOIN
#endif
#endif

				)
		{
			/* TODO we should add other types of constraints */

			/* Is the worker larger than requested ? */
			int worker_size = (int)_starpu_config.combined_workers[workerid - nworkers].worker_size;
			int worker0 = _starpu_config.combined_workers[workerid - nworkers].combined_workerid[0];
			return !!((worker_size <= task->cl->max_parallelism) &&
				_starpu_can_use_nth_implementation(_starpu_config.workers[worker0].arch, task->cl, nimpl) &&
				(!task->cl->can_execute || task->cl->can_execute(workerid, task, nimpl)));
		}
		else
		{
			/* We have a sequential task but a parallel worker */
			return 0;
		}
	}
}

/*
 * Runtime initialization methods
 */

static void _starpu_init_worker_queue(struct _starpu_worker *worker)
{
	_starpu_memory_node_register_condition(worker, &worker->sched_cond, worker->memory_node);
}

/*
 * Returns 0 if the given driver is one of the drivers that must be launched by
 * the application itself, and not by StarPU, 1 otherwise.
 */
static unsigned _starpu_may_launch_driver(struct starpu_conf *conf,
					  struct starpu_driver *d)
{
	if (conf->n_not_launched_drivers == 0 || conf->not_launched_drivers == NULL)
		return 1;

	/* Is <d> in conf->not_launched_drivers ? */
	unsigned i;
	for (i = 0; i < conf->n_not_launched_drivers; i++)
	{
		if (d->type != conf->not_launched_drivers[i].type)
			continue;

		/* Driver porters: adding your driver here is optional, only
		 * needed for supporting running the driver in a thread provided by
		 * the application. */

		switch (d->type)
		{
		case STARPU_CPU_WORKER:
			if (d->id.cpu_id == conf->not_launched_drivers[i].id.cpu_id)
				return 0;
			break;
		case STARPU_CUDA_WORKER:
			if (d->id.cuda_id == conf->not_launched_drivers[i].id.cuda_id)
				return 0;
			break;
		case STARPU_HIP_WORKER:
			if (d->id.hip_id == conf->not_launched_drivers[i].id.hip_id)
				return 0;
			break;
#ifdef STARPU_USE_OPENCL
		case STARPU_OPENCL_WORKER:
			if (d->id.opencl_id == conf->not_launched_drivers[i].id.opencl_id)
				return 0;
			break;
#endif
		default:
			STARPU_ABORT();
		}
	}

	return 1;
}

#ifdef STARPU_PERF_DEBUG
struct itimerval prof_itimer;
#endif

void _starpu_worker_init(struct _starpu_worker *workerarg, struct _starpu_machine_config *pconfig)
{
	workerarg->config = pconfig;
	STARPU_PTHREAD_MUTEX_INIT(&workerarg->mutex, NULL);
	/* arch initialized by topology.c */
	/* worker_mask initialized by topology.c */
	/* perf_arch initialized by topology.c */
	/* worker_thread initialized by _starpu_launch_drivers */
	/* devid initialized by topology.c */
	/* subworkerid initialized by topology.c */
	/* bindid initialized by topology.c */
	/* workerid initialized by topology.c */
	workerarg->combined_workerid = workerarg->workerid;
	workerarg->current_rank = 0;
	workerarg->worker_size = 1;
	STARPU_PTHREAD_COND_INIT(&workerarg->started_cond, NULL);
	STARPU_PTHREAD_COND_INIT(&workerarg->ready_cond, NULL);
	/* memory_node initialized by topology.c */
	STARPU_PTHREAD_COND_INIT(&workerarg->sched_cond, NULL);
	STARPU_PTHREAD_MUTEX_INIT(&workerarg->sched_mutex, NULL);
	starpu_task_prio_list_init(&workerarg->local_tasks);
	_starpu_ctx_change_list_init(&workerarg->ctx_change_list);
	workerarg->local_ordered_tasks = NULL;
	workerarg->local_ordered_tasks_size = 0;
	workerarg->current_ordered_task = 0;
	workerarg->current_ordered_task_order = 1;
	workerarg->current_task = NULL;
#ifdef STARPU_SIMGRID
	starpu_pthread_wait_init(&workerarg->wait);
	starpu_pthread_queue_register(&workerarg->wait, &_starpu_simgrid_task_queue[workerarg->workerid]);
#endif
	workerarg->task_transferring = NULL;
	workerarg->nb_buffers_transferred = 0;
	STARPU_HG_DISABLE_CHECKING(workerarg->nb_buffers_transferred);
	workerarg->nb_buffers_totransfer = 0;

	workerarg->first_task = 0;
	workerarg->ntasks = 0;
	/* set initialized by topology.c */
	workerarg->pipeline_length = 0;
	workerarg->pipeline_stuck = 0;
	workerarg->worker_is_running = 0;
	workerarg->worker_is_initialized = 0;
	workerarg->wait_for_worker_initialization = 0;
	workerarg->status = STATUS_INITIALIZING;
	workerarg->state_keep_awake = 0;
	/* name initialized by driver */
	/* short_name initialized by driver */
	workerarg->run_by_starpu = 1;
	workerarg->driver_ops = NULL;

	workerarg->sched_ctx_list = NULL;
	workerarg->tmp_sched_ctx = -1;
	workerarg->nsched_ctxs = 0;
	_starpu_barrier_counter_init(&workerarg->tasks_barrier, 0);

	workerarg->has_prev_init = 0;

	int ctx;
	for(ctx = 0; ctx < STARPU_NMAX_SCHED_CTXS; ctx++)
		workerarg->removed_from_ctx[ctx] = 0;

	workerarg->spinning_backoff = 1;

	for(ctx = 0; ctx < STARPU_NMAX_SCHED_CTXS; ctx++)
	{
		workerarg->shares_tasks_lists[ctx] = 0;
		workerarg->poped_in_ctx[ctx] = 0;
	}
	workerarg->reverse_phase[0] = 0;
	workerarg->reverse_phase[1] = 0;
	workerarg->pop_ctx_priority = 1;
	workerarg->is_slave_somewhere = 0;

	workerarg->state_relax_refcnt = 1;
#ifdef STARPU_SPINLOCK_CHECK
	workerarg->relax_on_file = __FILE__;
	workerarg->relax_on_line = __LINE__;
	workerarg->relax_on_func = __starpu_func__;
	workerarg->relax_off_file = NULL;
	workerarg->relax_off_line = 0;
	workerarg->relax_off_func = NULL;
#endif
	workerarg->state_sched_op_pending = 0;
	workerarg->state_changing_ctx_waiting = 0;
	workerarg->state_changing_ctx_notice = 0;
	workerarg->state_blocked_in_parallel_observed = 0;
	workerarg->state_blocked_in_parallel = 0;
	workerarg->state_block_in_parallel_req = 0;
	workerarg->state_block_in_parallel_ack = 0;
	workerarg->state_unblock_in_parallel_req = 0;
	workerarg->state_unblock_in_parallel_ack = 0;
	workerarg->block_in_parallel_ref_count = 0;
	_starpu_perf_counter_sample_init(&workerarg->perf_counter_sample, starpu_perf_counter_scope_per_worker);
	workerarg->enable_knob = 1;
	workerarg->bindid_requested = -1;

	/* cpu_set/hwloc_cpu_set/hwloc_obj initialized in topology.c */
}

static void _starpu_worker_deinit(struct _starpu_worker *workerarg)
{
	(void) workerarg;

#ifdef STARPU_SIMGRID
	starpu_pthread_queue_unregister(&workerarg->wait, &_starpu_simgrid_task_queue[workerarg->workerid]);
	starpu_pthread_wait_destroy(&workerarg->wait);
#endif
	_starpu_perf_counter_sample_exit(&workerarg->perf_counter_sample);
}

void _starpu_driver_start(struct _starpu_worker *worker, enum starpu_worker_archtype archtype, unsigned sync STARPU_ATTRIBUTE_UNUSED)
{
	(void) archtype;
	int devid = worker->devid;
	(void) devid;

	_starpu_trace_register_thread(worker->bindid);
	_starpu_trace_worker_init_start(worker, archtype, sync);

	_starpu_set_local_worker_key(worker);

	STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
	worker->worker_is_running = 1;
	STARPU_PTHREAD_COND_SIGNAL(&worker->started_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);

	_starpu_bind_thread_on_cpu(worker->bindid, worker->workerid, NULL);

#if defined(STARPU_PERF_DEBUG) && !defined(STARPU_SIMGRID)
	setitimer(ITIMER_PROF, &prof_itimer, NULL);
#endif

	_STARPU_DEBUG("worker %p %d for dev %d is ready on logical cpu %d\n", worker, worker->workerid, devid, worker->bindid);
#ifdef STARPU_HAVE_HWLOC
	_STARPU_DEBUG("worker %p %d cpuset start at %d\n", worker, worker->workerid, hwloc_bitmap_first(worker->hwloc_cpu_set));
#endif
}

static void _starpu_launch_drivers(struct _starpu_machine_config *pconfig)
{
	pconfig->running = 1;
	pconfig->pause_depth = 0;
	pconfig->submitting = 1;
	STARPU_HG_DISABLE_CHECKING(pconfig->watchdog_ok);

	unsigned nworkers = pconfig->topology.nworkers;
	unsigned worker;

#if defined(STARPU_PERF_DEBUG) && !defined(STARPU_SIMGRID)
	/* Get itimer of the main thread, to set it for the worker threads */
	getitimer(ITIMER_PROF, &prof_itimer);
#endif
	STARPU_AYU_INIT();

	/* Launch workers asynchronously */
	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &pconfig->workers[worker];
		workerarg->wait_for_worker_initialization = 0;

		_STARPU_DEBUG("initialising worker %u/%u\n", worker, nworkers);

		_starpu_init_worker_queue(workerarg);

		struct _starpu_worker_set *worker_set = workerarg->set;
		struct _starpu_worker_set *driver_worker_set = workerarg->driver_worker_set;

		/* For worker sets, we only start a thread for the first worker.  */
		if (!worker_set || worker_set->workers == workerarg)
		{
			struct starpu_driver driver;

			if (worker_set)
			{
				worker_set->set_is_initialized = 0;
				worker_set->wait_for_set_initialization = !driver_worker_set || driver_worker_set == worker_set;
			}

			workerarg->driver_ops = starpu_driver_info[workerarg->arch].driver_ops;
			workerarg->wait_for_worker_initialization = starpu_driver_info[workerarg->arch].wait_for_worker_initialization;

			if (workerarg->driver_ops)
			{
				driver.type = workerarg->arch;
				if (workerarg->driver_ops->set_devid)
					workerarg->driver_ops->set_devid(&driver, workerarg);
			}

			if ((driver_worker_set && driver_worker_set != worker_set) ||
			    (workerarg->driver_ops && !_starpu_may_launch_driver(&pconfig->conf, &driver)))
				workerarg->run_by_starpu = 0;
		}
		else
			workerarg->run_by_starpu = 0;
	}

	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &pconfig->workers[worker];
		struct _starpu_worker_set *worker_set = workerarg->set;

		/* For worker sets, we only start a thread for the first worker.  */
		if (workerarg->run_by_starpu)
		{
			starpu_pthread_t *worker_thread;

			if (worker_set)
				worker_thread = &worker_set->worker_thread;
			else
				worker_thread = &workerarg->worker_thread;

			/* For driver worker sets, we only start a thread for the first worker set.  */
			STARPU_PTHREAD_CREATE_ON(
				starpu_driver_info[workerarg->arch].name_upper,
				worker_thread,
				NULL,
				starpu_driver_info[workerarg->arch].run_worker,
				workerarg,
				_starpu_simgrid_get_host_by_worker(workerarg));
		}

#ifdef STARPU_USE_FXT
		/* In tracing mode, make sure the thread is really started
		 * before starting another one, to make sure they appear in
		 * order in the trace.
		 */
		if (fut_active && workerarg->run_by_starpu)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&workerarg->mutex);
			while (!workerarg->worker_is_running)
				STARPU_PTHREAD_COND_WAIT(&workerarg->started_cond, &workerarg->mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&workerarg->mutex);
		}
#endif
	}

	for (worker = 0; worker < nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &pconfig->workers[worker];

		_STARPU_DEBUG("waiting for worker %u initialization\n", worker);
		if (!workerarg->run_by_starpu)
			continue;

		struct _starpu_worker_set *worker_set = workerarg->set;

		if (worker_set && worker_set->wait_for_set_initialization == 1)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
			while (!worker_set->set_is_initialized)
				STARPU_PTHREAD_COND_WAIT(&worker_set->ready_cond,
							 &worker_set->mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);
			worker_set->started = 1;
			worker_set->wait_for_set_initialization = 0;
		}
		else if (workerarg->wait_for_worker_initialization == 1)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&workerarg->mutex);
			while (!workerarg->worker_is_initialized)
				STARPU_PTHREAD_COND_WAIT(&workerarg->ready_cond, &workerarg->mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&workerarg->mutex);
			workerarg->wait_for_worker_initialization = 0;
		}
	}

	_STARPU_DEBUG("finished launching drivers\n");
}

void starpu_worker_wait_for_initialisation()
{
	unsigned nworkers = starpu_worker_get_count();
	unsigned workerid;

	for (workerid = 0; workerid < nworkers; workerid++)
	{
		struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);

		_STARPU_DEBUG("waiting for worker %u initialization\n", workerid);
		if (!worker->run_by_starpu)
			break;

		struct _starpu_worker_set *worker_set = worker->set;

		if (worker_set)
		{
			STARPU_PTHREAD_MUTEX_LOCK(&worker_set->mutex);
			while (!worker_set->set_is_initialized)
				STARPU_PTHREAD_COND_WAIT(&worker_set->ready_cond,
							 &worker_set->mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&worker_set->mutex);
		}
		else
		{
			STARPU_PTHREAD_MUTEX_LOCK(&worker->mutex);
			while (!worker->worker_is_initialized)
				STARPU_PTHREAD_COND_WAIT(&worker->ready_cond, &worker->mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&worker->mutex);
		}
	}
}

/* Initialize the starpu_conf with default values */
int starpu_conf_init(struct starpu_conf *conf)
{
	if (!conf)
		return -EINVAL;

	memset(conf, 0, sizeof(*conf));
	conf->magic = 42;
	conf->will_use_mpi = 0;
	conf->sched_policy_name = starpu_getenv("STARPU_SCHED");
	conf->sched_policy = NULL;
	conf->global_sched_ctx_min_priority = starpu_getenv_number("STARPU_MIN_PRIO");
	conf->global_sched_ctx_max_priority = starpu_getenv_number("STARPU_MAX_PRIO");
	conf->catch_signals = starpu_getenv_number_default("STARPU_CATCH_SIGNALS", 1);

	/* Note that starpu_getenv_number returns -1 in case the variable is
	 * not defined */
	/* Backward compatibility: check the value of STARPU_NCPUS if
	 * STARPU_NCPU is not set. */
	conf->ncpus = starpu_getenv_number("STARPU_NCPU");
	if (conf->ncpus == -1)
		conf->ncpus = starpu_getenv_number("STARPU_NCPUS");
	conf->reserve_ncpus = starpu_getenv_number("STARPU_RESERVE_NCPU");
	conf->ncuda = starpu_getenv_number("STARPU_NCUDA");
	conf->nhip = starpu_getenv_number("STARPU_NHIP");
	conf->nopencl = starpu_getenv_number("STARPU_NOPENCL");
	conf->nmax_fpga = starpu_getenv_number("STARPU_NMAX_FPGA");
	conf->nmpi_ms = starpu_getenv_number("STARPU_NMPI_MS");
	conf->ntcpip_ms = starpu_getenv_number("STARPU_NTCPIP_MS");
	conf->calibrate = starpu_getenv_number("STARPU_CALIBRATE");
	conf->bus_calibrate = starpu_getenv_number("STARPU_BUS_CALIBRATE");

	if (conf->calibrate == -1)
	     conf->calibrate = 0;

	if (conf->bus_calibrate == -1)
	     conf->bus_calibrate = 0;

	conf->use_explicit_workers_bindid = 0; /* TODO */
	conf->use_explicit_workers_cuda_gpuid = 0; /* TODO */
	conf->use_explicit_workers_hip_gpuid = 0; /* TODO */
	conf->use_explicit_workers_opencl_gpuid = 0; /* TODO */
	conf->use_explicit_workers_max_fpga_deviceid = 0; /* TODO */
	conf->use_explicit_workers_mpi_ms_deviceid = 0; /* TODO */

	conf->single_combined_worker = starpu_getenv_number("STARPU_SINGLE_COMBINED_WORKER");
	if (conf->single_combined_worker == -1)
	     conf->single_combined_worker = 0;

#if defined(STARPU_DATA_LOCALITY_ENFORCE)
	conf->data_locality_enforce = 1;
#else
	conf->data_locality_enforce = starpu_getenv_number("STARPU_DATA_LOCALITY_ENFORCE");
	if (conf->data_locality_enforce == -1)
		conf->data_locality_enforce = 0;
#endif

#if defined(STARPU_DISABLE_ASYNCHRONOUS_COPY)
	conf->disable_asynchronous_copy = 1;
#else
	conf->disable_asynchronous_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_COPY");
	if (conf->disable_asynchronous_copy == -1)
		conf->disable_asynchronous_copy = 0;
#endif

#if defined(STARPU_DISABLE_ASYNCHRONOUS_CUDA_COPY)
	conf->disable_asynchronous_cuda_copy = 1;
#else
	conf->disable_asynchronous_cuda_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_CUDA_COPY");
	if (conf->disable_asynchronous_cuda_copy == -1)
		conf->disable_asynchronous_cuda_copy = 0;
#endif


#if defined(STARPU_DISABLE_ASYNCHRONOUS_HIP_COPY)
	conf->disable_asynchronous_hip_copy = 1;
#else
	conf->disable_asynchronous_hip_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_HIP_COPY");
	if (conf->disable_asynchronous_hip_copy == -1)
		conf->disable_asynchronous_hip_copy = 0;
#endif

#if defined(STARPU_DISABLE_ASYNCHRONOUS_OPENCL_COPY)
	conf->disable_asynchronous_opencl_copy = 1;
#else
	conf->disable_asynchronous_opencl_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_OPENCL_COPY");
	if (conf->disable_asynchronous_opencl_copy == -1)
		conf->disable_asynchronous_opencl_copy = 0;
#endif

#if defined(STARPU_DISABLE_ASYNCHRONOUS_MAX_FPGA_COPY)
	conf->disable_asynchronous_max_fpga_copy = 1;
#else
	conf->disable_asynchronous_max_fpga_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_MAX_FPGA_COPY");
	if (conf->disable_asynchronous_max_fpga_copy == -1)
		conf->disable_asynchronous_max_fpga_copy = 0;
#endif

#if defined(STARPU_DISABLE_ASYNCHRONOUS_MPI_MS_COPY)
	conf->disable_asynchronous_mpi_ms_copy = 1;
#else
	conf->disable_asynchronous_mpi_ms_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_MPI_MS_COPY");
	if(conf->disable_asynchronous_mpi_ms_copy == -1)
		conf->disable_asynchronous_mpi_ms_copy = 0;
#endif

#if defined(STARPU_DISABLE_ASYNCHRONOUS_TCPIP_MS_COPY)
	conf->disable_asynchronous_tcpip_ms_copy = 1;
#else
	conf->disable_asynchronous_tcpip_ms_copy = starpu_getenv_number("STARPU_DISABLE_ASYNCHRONOUS_TCPIP_MS_COPY");
	if(conf->disable_asynchronous_tcpip_ms_copy == -1)
		conf->disable_asynchronous_tcpip_ms_copy = 0;
#endif

	conf->enable_map = starpu_getenv_number("STARPU_ENABLE_MAP");
	if (conf->enable_map == -1)
		conf->enable_map = 0;

	/* 64MiB by default */
	conf->trace_buffer_size = ((uint64_t) starpu_getenv_number_default("STARPU_TRACE_BUFFER_SIZE", 64)) << 20;

	conf->driver_spinning_backoff_min = (unsigned) starpu_getenv_number_default("STARPU_BACKOFF_MIN", 1);
	conf->driver_spinning_backoff_max = (unsigned) starpu_getenv_number_default("STARPU_BACKOFF_MAX", 32);

	/* Do not start performance counter collection by default */
	conf->start_perf_counter_collection = 0;

	conf->cuda_only_fast_alloc_other_memnodes = starpu_getenv_number_default("STARPU_CUDA_ONLY_FAST_ALLOC_OTHER_MEMNODES", 0);
	return 0;
}

int starpu_conf_noworker(struct starpu_conf *conf)
{
	conf->ncpus = 0;
	conf->ncuda = 0;
	conf->nhip = 0;
	conf->nopencl = 0;
	conf->nmax_fpga = 0;
	conf->nmpi_ms = 0;
	conf->ntcpip_ms = 0;
	return 0;
}

static void _starpu_conf_set_value_against_environment(char *name, int *value, int precedence_over_env)
{
	if (precedence_over_env == 0)
	{
		int number;
		number = starpu_getenv_number(name);
		if (number != -1)
		{
			*value = number;
		}
	}
}

void _starpu_conf_check_environment(struct starpu_conf *conf)
{
	char *sched = starpu_getenv("STARPU_SCHED");
	if (sched)
	{
		conf->sched_policy_name = sched;
	}

	_starpu_conf_set_value_against_environment("STARPU_NCPUS", &conf->ncpus, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_NCPU", &conf->ncpus, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_RESERVE_NCPU", &conf->reserve_ncpus, conf->precedence_over_environment_variables);
	int main_thread_bind = starpu_getenv_number_default("STARPU_MAIN_THREAD_BIND", 0);
	if (main_thread_bind)
	{
		/* Reserve a core for main */
		if (conf->reserve_ncpus == -1)
			conf->reserve_ncpus = 1;
		else
			conf->reserve_ncpus++;
	}
	_starpu_conf_set_value_against_environment("STARPU_NCUDA", &conf->ncuda, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_NHIP", &conf->nhip, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_NOPENCL", &conf->nopencl, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_NMAX_FPGA", &conf->nmax_fpga, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_NMPI_MS", &conf->nmpi_ms, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_NTCPIP_MS", &conf->ntcpip_ms, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_CALIBRATE", &conf->calibrate, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_BUS_CALIBRATE", &conf->bus_calibrate, conf->precedence_over_environment_variables);
#ifdef STARPU_SIMGRID
	if (conf->calibrate == 2)
	{
		_STARPU_DISP("Warning: History will be cleared due to calibrate or STARPU_CALIBRATE being set to 2. This will prevent simgrid from having task simulation times!\n");
	}
	if (conf->bus_calibrate)
	{
		_STARPU_DISP("Warning: Bus calibration will be cleared due to bus_calibrate or STARPU_BUS_CALIBRATE being set. This will prevent simgrid from having data transfer simulation times!\n");
	}
#endif
	_starpu_conf_set_value_against_environment("STARPU_SINGLE_COMBINED_WORKER", &conf->single_combined_worker, conf->precedence_over_environment_variables);

	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_COPY", &conf->disable_asynchronous_copy, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_CUDA_COPY", &conf->disable_asynchronous_cuda_copy, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_HIP_COPY", &conf->disable_asynchronous_hip_copy, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_OPENCL_COPY", &conf->disable_asynchronous_opencl_copy, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_MAX_FPGA_COPY", &conf->disable_asynchronous_max_fpga_copy, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_MPI_MS_COPY", &conf->disable_asynchronous_mpi_ms_copy, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_DISABLE_ASYNCHRONOUS_TCPIP_MS_COPY", &conf->disable_asynchronous_tcpip_ms_copy, conf->precedence_over_environment_variables);

	_starpu_conf_set_value_against_environment("STARPU_ENABLE_MAP", &conf->enable_map, conf->precedence_over_environment_variables);

	asynchronous_copy_disabled[STARPU_CPU_RAM] = 0;
	asynchronous_copy_disabled[STARPU_CUDA_RAM] = conf->disable_asynchronous_cuda_copy;
	asynchronous_copy_disabled[STARPU_HIP_RAM] = conf->disable_asynchronous_hip_copy;
	asynchronous_copy_disabled[STARPU_OPENCL_RAM] = conf->disable_asynchronous_opencl_copy;
	asynchronous_copy_disabled[STARPU_MAX_FPGA_RAM] = conf->disable_asynchronous_max_fpga_copy;
	asynchronous_copy_disabled[STARPU_DISK_RAM] = 0;
	asynchronous_copy_disabled[STARPU_MPI_MS_RAM] = conf->disable_asynchronous_mpi_ms_copy;
	asynchronous_copy_disabled[STARPU_TCPIP_MS_RAM] = conf->disable_asynchronous_tcpip_ms_copy;

	_starpu_conf_set_value_against_environment("STARPU_MIN_PRIO", &conf->global_sched_ctx_min_priority, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_MAX_PRIO", &conf->global_sched_ctx_max_priority, conf->precedence_over_environment_variables);
	_starpu_conf_set_value_against_environment("STARPU_CATCH_SIGNALS", &conf->catch_signals, conf->precedence_over_environment_variables);
}

struct starpu_tree* starpu_workers_get_tree(void)
{
	return _starpu_config.topology.tree;
}

#ifdef STARPU_HAVE_HWLOC
#if HWLOC_API_VERSION >= 0x20000
#define NORMAL_CHILD(obj) 1
#else
#define NORMAL_CHILD(obj) ((obj)->type < HWLOC_OBJ_BRIDGE)
#endif

static void _fill_tree(struct starpu_tree *tree, hwloc_obj_t curr_obj, unsigned depth, hwloc_topology_t topology, struct starpu_tree *parent)
{
	unsigned i, j;
	unsigned arity;
#if HWLOC_API_VERSION >= 0x20000
	arity = curr_obj->arity;
#else
	arity = 0;
	for(i = 0; i < curr_obj->arity; i++)
	{
		if (!NORMAL_CHILD(curr_obj->children[i]))
			/* I/O stuff, stop caring */
			break;
		arity++;
	}
#endif

	if (arity == 1)
	{
		/* Nothing interestin here, skip level */
		_fill_tree(tree, curr_obj->children[0], depth+1, topology, parent);
		return;
	}

	starpu_tree_insert(tree, curr_obj->logical_index, depth, curr_obj->type == HWLOC_OBJ_PU, arity, parent);
	starpu_tree_prepare_children(arity, tree);
	j = 0;
	for(i = 0; i < arity; i++)
	{
		hwloc_obj_t child = curr_obj->children[i];
		if (!NORMAL_CHILD(child))
			/* I/O stuff, stop caring (shouldn't happen, though) */
			break;
#if 0
		char string[128];
		hwloc_obj_snprintf(string, sizeof(string), topology, child, "#", 0);
		printf("%*s%s %d is_pu %d \n", 0, "", string, child->logical_index, child->type == HWLOC_OBJ_PU);
#endif
		_fill_tree(&tree->nodes[j], child, depth+1, topology, tree);
		j++;
	}
}
#endif

static void _starpu_build_tree(void)
{
#ifdef STARPU_HAVE_HWLOC
	struct starpu_tree *tree;
	_STARPU_MALLOC(tree, sizeof(struct starpu_tree));
	_starpu_config.topology.tree = tree;

	hwloc_obj_t root = hwloc_get_root_obj(_starpu_config.topology.hwtopology);

#if 0
	char string[128];
	hwloc_obj_snprintf(string, sizeof(string), topology, root, "#", 0);
	printf("%*s%s %d is_pu = %d \n", 0, "", string, root->logical_index, root->type == HWLOC_OBJ_PU);
#endif

	/* level, is_pu, is in the tree (it will be true only after add) */
	_fill_tree(tree, root, 0, _starpu_config.topology.hwtopology, NULL);
#endif
}

typedef void (*hook_func_t)(void);
static hook_func_t _hook_funcs[10];
static int _hook_func_nb=0;

void _starpu_crash_add_hook(void (*hook_func)(void))
{
	STARPU_ASSERT_MSG(_hook_func_nb < 10, "The number of crash funcs has exceeded the limit\n");
	_hook_funcs[_hook_func_nb] = hook_func;
	_hook_func_nb++;
}

void _starpu_crash_call_hooks()
{
	int i;

	/*_STARPU_DISP("Time: %f\n", starpu_timing_now());*/
	for(i=0 ; i<_hook_func_nb; i++)
		_hook_funcs[i]();
}

static starpu_pthread_mutex_t sig_handlers_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static void (*act_sigint)(int);
static void (*act_sigsegv)(int);
static void (*act_sigabrt)(int);
#ifdef SIGTRAP
static void (*act_sigtrap)(int);
#endif

void _starpu_handler(int sig)
{
#ifdef STARPU_VERBOSE
	_STARPU_MSG("Catching signal '%d'\n", sig);
#endif
	_starpu_trace_finalize();
	if (sig == SIGINT)
	{
		void (*sig_act)(int) = act_sigint;
		if (sig_act == NULL)
			sig_act = SIG_DFL;
		signal(SIGINT, sig_act);
	}
	if (sig == SIGSEGV)
	{
		STARPU_DUMP_BACKTRACE();
		void (*sig_act)(int) = act_sigsegv;
		if (sig_act == NULL)
			sig_act = SIG_DFL;
		signal(SIGSEGV, sig_act);
	}
	if (sig == SIGABRT)
	{
		void (*sig_act)(int) = act_sigabrt;
		if (sig_act == NULL)
			sig_act = SIG_DFL;
		signal(SIGABRT, sig_act);
	}
#ifdef SIGTRAP
	if (sig == SIGTRAP)
	{
		void (*sig_act)(int) = act_sigtrap;
		if (sig_act == NULL)
			sig_act = SIG_DFL;
		signal(SIGTRAP, sig_act);
	}
#endif

	_starpu_crash_call_hooks();
#ifdef STARPU_VERBOSE
	_STARPU_MSG("Rearming signal '%d'\n", sig);
#endif
	raise(sig);
}

void _starpu_catch_signals(void)
{
	if (_starpu_config.conf.catch_signals == 1)
	{
		static void (*old_sig_act)(int);
		old_sig_act  = signal(SIGINT, _starpu_handler);
		if (old_sig_act != _starpu_handler)
			act_sigint  = old_sig_act;

		old_sig_act = signal(SIGSEGV, _starpu_handler);
		if (old_sig_act != _starpu_handler)
			act_sigsegv  = old_sig_act;

		old_sig_act = signal(SIGABRT, _starpu_handler);
		if (old_sig_act != _starpu_handler)
			act_sigabrt  = old_sig_act;

#ifdef SIGTRAP
		old_sig_act = signal(SIGTRAP, _starpu_handler);
		if (old_sig_act != _starpu_handler)
			act_sigtrap  = old_sig_act;
#endif
	}
	else
	{
		if (act_sigint != NULL)
		{
			signal(SIGINT, act_sigint);
			act_sigint = NULL;
		}

		if (act_sigsegv != NULL)
		{
			signal(SIGSEGV, act_sigsegv);
			act_sigsegv = NULL;
		}

		if (act_sigabrt != NULL)
		{
			signal(SIGABRT, act_sigsegv);
			act_sigabrt = NULL;
		}

#ifdef SIGTRAP
		if (act_sigtrap != NULL)
		{
			signal(SIGTRAP, act_sigtrap);
			act_sigtrap = NULL;
		}
#endif
	}
}

void _starpu_set_catch_signals(int do_catch_signal)
{
	STARPU_PTHREAD_MUTEX_LOCK(&sig_handlers_mutex);
	_starpu_config.conf.catch_signals = do_catch_signal;
	_starpu_catch_signals();
	STARPU_PTHREAD_MUTEX_UNLOCK(&sig_handlers_mutex);
}

int _starpu_get_catch_signals(void)
{
	return _starpu_config.conf.catch_signals;
}

void starpu_drivers_preinit(void)
{
	_starpu_cpu_preinit();
	_starpu_cuda_preinit();
	_starpu_hip_preinit();
	_starpu_opencl_preinit();
	_starpu_max_fpga_preinit();
	_starpu_mpi_ms_preinit();
	_starpu_tcpip_ms_preinit();
	_starpu_disk_preinit();
}

int starpu_init(struct starpu_conf *user_conf)
{
	return starpu_initialize(user_conf, NULL, NULL);
}

int starpu_initialize(struct starpu_conf *user_conf, int *argc, char ***argv)
{
	int is_a_sink = 0; /* Always defined. If the MP infrastructure is not
			    * used, we cannot be a sink. */
	unsigned worker;

#if !defined(STARPU_SIMGRID) && !defined(STARPU_USE_MP)
	(void)argc;
	(void)argv;
#endif

	STARPU_HG_DISABLE_CHECKING(_starpu_worker_parallel_blocks);
#ifdef STARPU_SIMGRID
	/* This initializes the simgrid thread library, thus needs to be early */
	_starpu_simgrid_init_early(argc, argv);
#endif
#ifdef STARPU_NOSV
	{
		int status = nosv_init();
		if (status != 0)
		{
			_STARPU_DISP("nOS-V: nosv_init failed with error %d\n", status);
			return -ENODEV;
		}
		_STARPU_DISP("nOS-V: nosv_init called succesfully\n");
	}
#endif

	STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
	while (initialized == CHANGING)
		/* Wait for the other one changing it */
		STARPU_PTHREAD_COND_WAIT(&init_cond, &init_mutex);
	init_count++;
	if (initialized == INITIALIZED)
	{
		/* He initialized it, don't do it again, and let the others get the mutex */
		STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);
		return 0;
	}
	/* initialized == UNINITIALIZED */
	initialized = CHANGING;
	STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);

	/* This initializes _starpu_silent, thus needs to be early */
	_starpu_util_init();

	int rc = _starpu_prof_tool_try_load();
	(void) rc; /* unused for now */
	int no_ms = 0;

#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_init_begin, 0, 0, starpu_prof_tool_driver_cpu, -1, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_init(&pi, NULL, NULL);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_init_begin(&pi, NULL, NULL);
#endif

#ifdef STARPU_USE_MP
	_starpu_set_argc_argv(argc, argv);

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	if (_starpu_mpi_common_mp_init() == -ENODEV)
	{
		_STARPU_DISP("Warning: starpu_init called several times, ignoring slaves in this restart\n");
		no_ms = 1;
	}
#endif

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	if (_starpu_tcpip_common_mp_init() == -ENODEV)
	{
		_STARPU_DISP("Warning: starpu_init called several times, ignoring slaves in this restart\n");
		no_ms = 1;
	}
#endif

	/* If StarPU was configured to use MP sinks, we have to control the
	 * kind on node we are running on : host or sink ? */
	if (starpu_getenv("STARPU_SINK"))
		is_a_sink = 1;
#endif /* STARPU_USE_MP */

	int ret;

#ifdef STARPU_OPENMP
	_starpu_omp_dummy_init();
#endif

#ifdef STARPU_SIMGRID
	/* Warn when the lots of stacks malloc()-ated by simgrid for transfer
	 * processes will take a long time to get initialized */
	char *perturb = starpu_getenv("MALLOC_PERTURB_");
	if (perturb && perturb[0] && atoi(perturb) != 0)
		_STARPU_DISP("Warning: MALLOC_PERTURB_ is set to non-zero, this makes simgrid run very slow\n");
#else
#ifdef __GNUC__
#ifndef __OPTIMIZE__
	_STARPU_DISP("Warning: StarPU was configured with --enable-debug (-O0), and is thus not optimized\n");
#endif
#endif
#ifdef STARPU_SPINLOCK_CHECK
	_STARPU_DISP("Warning: StarPU was configured with --enable-spinlock-check, which slows down a bit\n");
#endif
#if 0
#ifndef STARPU_NO_ASSERT
	_STARPU_DISP("Warning: StarPU was configured without --enable-fast\n");
#endif
#endif
#ifdef STARPU_MEMORY_STATS
	_STARPU_DISP("Warning: StarPU was configured with --enable-memory-stats, which slows down a bit\n");
#endif
#ifdef STARPU_VERBOSE
	_STARPU_DISP("Warning: StarPU was configured with --enable-verbose, which slows down a bit\n");
#endif
#ifdef STARPU_USE_FXT
	if (starpu_fxt_is_enabled())
		_STARPU_DISP("Warning: FxT is enabled, which slows down a bit, limits scalability and makes worker initialization sequential\n");
#else
	if (starpu_getenv_number("STARPU_FXT_TRACE") > 0)
		_STARPU_DISP("Warning: FxT trace is requested but StarPU was configured without FxT support\n");
#endif
#ifdef STARPU_FXT_LOCK_TRACES
	_STARPU_DISP("Warning: StarPU was configured with --enable-fxt-lock, which slows down things a huge lot, and is really only meant for StarPU insides debugging. Did you really want to enable that?\n");
#endif
#ifdef STARPU_PERF_DEBUG
	_STARPU_DISP("Warning: StarPU was configured with --enable-perf-debug, which slows down a bit\n");
#endif
#ifdef STARPU_MODEL_DEBUG
	_STARPU_DISP("Warning: StarPU was configured with --enable-model-debug, which slows down a bit\n");
#endif
#ifdef __linux__
	{
		struct utsname buf;
		if (uname(&buf) == 0
		 && (!strncmp(buf.release, "4.7.", 4)
		  || !strncmp(buf.release, "4.8.", 4)))
			_STARPU_DISP("Warning: This system is running a 4.7 or 4.8 kernel. These have a severe scheduling performance regression issue, please upgrade to at least 4.9.\n");
	}
#endif
#endif

	if (starpu_getenv("STARPU_ENABLE_STATS"))
	{
		_STARPU_DISP("Warning: STARPU_ENABLE_STATS is enabled, which slows down a bit\n");
	}

#ifndef STARPU_SIMGRID
	if (starpu_getenv_number_default("STARPU_SIMGRID", 0))
	{
		_STARPU_DISP("Simulation mode requested, but this libstarpu was built without simgrid support, please recompile\n");
		STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
		init_count--;
		initialized = UNINITIALIZED;
		/* Let somebody else try to do it */
		STARPU_PTHREAD_COND_SIGNAL(&init_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);
		return -EINVAL;
	}
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
	WSADATA wsadata;
	WSAStartup(MAKEWORD(1,0), &wsadata);
#endif

	STARPU_AYU_PREINIT();
	/* store the pointer to the user explicit configuration during the
	 * initialization */
	if (user_conf == NULL)
		 starpu_conf_init(&_starpu_config.conf);
	else
	{
		if (user_conf->magic != 42)
		{
			_STARPU_DISP("starpu_conf structure needs to be initialized with starpu_conf_init\n");
			STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
			init_count--;
			initialized = UNINITIALIZED;
			/* Let somebody else try to do it */
			STARPU_PTHREAD_COND_SIGNAL(&init_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);
			return -EINVAL;
		}
		_starpu_config.conf = *user_conf;
	}
	_starpu_conf_check_environment(&_starpu_config.conf);

	if (no_ms)
	{
		_starpu_config.conf.nmpi_ms = 0;
		_starpu_config.conf.ntcpip_ms = 0;
	}

	if (is_a_sink && _starpu_config.conf.nmpi_ms == 0 &&
		_starpu_config.conf.ntcpip_ms == 0)
	{
		/* MS was explicitly disabled, abort sinks and leave source alone */

#ifdef STARPU_USE_MPI_MASTER_SLAVE
		if (_starpu_mpi_common_is_mp_initialized())
			_starpu_mpi_common_mp_deinit();
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		if (_starpu_tcpip_common_is_mp_initialized())
			_starpu_tcpip_common_mp_deinit();
#endif

		STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
		init_count--;
		initialized = UNINITIALIZED;
		/* Let somebody else try to do it */
		STARPU_PTHREAD_COND_SIGNAL(&init_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);
		return -ENODEV;
	}

	/* Make a copy of arrays */
	if (_starpu_config.conf.sched_policy_name)
		_starpu_config.conf.sched_policy_name = strdup(_starpu_config.conf.sched_policy_name);
	if (_starpu_config.conf.n_cuda_opengl_interoperability)
	{
		size_t size = _starpu_config.conf.n_cuda_opengl_interoperability * sizeof(*_starpu_config.conf.cuda_opengl_interoperability);
		unsigned *copy;
		_STARPU_MALLOC(copy, size);
		memcpy(copy, _starpu_config.conf.cuda_opengl_interoperability, size);
		_starpu_config.conf.cuda_opengl_interoperability = copy;
	}
	if (_starpu_config.conf.n_not_launched_drivers)
	{
		size_t size = _starpu_config.conf.n_not_launched_drivers * sizeof(*_starpu_config.conf.not_launched_drivers);
		struct starpu_driver *copy;
		_STARPU_MALLOC(copy, size);
		memcpy(copy, _starpu_config.conf.not_launched_drivers, size);
		_starpu_config.conf.not_launched_drivers = copy;
	}

	_hook_func_nb = 0;

	/* Let drivers register themselves */
	starpu_drivers_preinit();

	_starpu_sched_init();
	_starpu_job_init();
	_starpu_graph_init();

	_starpu_init_all_sched_ctxs(&_starpu_config);
	_starpu_init_progression_hooks();
	_starpu_init_idle_hooks();

	_starpu_init_tags();

	_starpu_init_perfmodel();

	/* Initialize the profiling tool(s) */

	_starpu_trace_initialize();

	_starpu_open_debug_logfile();

	_starpu_data_interface_init();

	_starpu_timing_init();

	_starpu_load_bus_performance_files();

	/* Note: nothing before here should be allocating anything, in case we
	 * actually return ENODEV here */

	/* Depending on whether we are a MP sink or not, we must build the
	 * topology with MP nodes or not. */
	ret = _starpu_build_topology(&_starpu_config, is_a_sink);
	/* sink doesn't exit even if no worker discovered */
	if (ret && !is_a_sink)
	{
		starpu_perfmodel_free_sampling();
		STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
		init_count--;

		_starpu_destroy_machine_config(&_starpu_config, is_a_sink);

		free((char*) _starpu_config.conf.sched_policy_name);
		if (_starpu_config.conf.n_cuda_opengl_interoperability)
			free(_starpu_config.conf.cuda_opengl_interoperability);
		if (_starpu_config.conf.n_not_launched_drivers)
			free(_starpu_config.conf.not_launched_drivers);

#ifdef STARPU_USE_MPI_MASTER_SLAVE
		if (_starpu_mpi_common_is_mp_initialized())
			_starpu_mpi_common_mp_deinit();
#endif
#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
		if (_starpu_tcpip_common_is_mp_initialized())
			_starpu_tcpip_common_mp_deinit();
#endif

		initialized = UNINITIALIZED;
		/* Let somebody else try to do it */
		STARPU_PTHREAD_COND_SIGNAL(&init_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);

		_starpu_trace_finalize();
		return ret;
	}

	_starpu_profiling_init();

	_starpu_task_init();

	for (worker = 0; worker < _starpu_config.topology.nworkers; worker++)
		_starpu_worker_init(&_starpu_config.workers[worker], &_starpu_config);

//FIXME: find out if the variable STARPU_CHECK_ENTIRE_PLATFORM is really needed, for now, just set 1 as a default value
	check_entire_platform = 1;//starpu_getenv_number("STARPU_CHECK_ENTIRE_PLATFORM");

	_starpu_config.disable_kernels = starpu_getenv_number("STARPU_DISABLE_KERNELS");
	STARPU_PTHREAD_KEY_CREATE(&_starpu_worker_key, NULL);
	STARPU_PTHREAD_KEY_CREATE(&_starpu_worker_set_key, NULL);
	_starpu_keys_initialized = 1;
	STARPU_WMB();

	if (!is_a_sink)
	{
		_starpu_build_tree();

		struct starpu_sched_policy *selected_policy = _starpu_select_sched_policy(&_starpu_config, _starpu_config.conf.sched_policy_name);
		_starpu_create_sched_ctx(selected_policy, NULL, -1, 1, "init", (_starpu_config.conf.global_sched_ctx_min_priority != -1), _starpu_config.conf.global_sched_ctx_min_priority, (_starpu_config.conf.global_sched_ctx_max_priority != -1), _starpu_config.conf.global_sched_ctx_max_priority, 1, _starpu_config.conf.sched_policy_callback, NULL,  0, NULL, 0);
	}

	_starpu_initialize_registered_performance_models();
	_starpu_perf_counter_init(&_starpu_config);
	_starpu_perf_knob_init();

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	_starpu_cuda_init();
#endif

#if defined(STARPU_USE_HIP)
	_starpu_hip_init();
#endif

#ifdef STARPU_SIMGRID
	_starpu_simgrid_init();
#endif
	if (!is_a_sink)
	{
		/* Launch "basic" workers (ie. non-combined workers) */
		_starpu_launch_drivers(&_starpu_config);
		/* Allocate swap, if any */
		_starpu_swap_init();
	}

	_starpu_watchdog_init();

	_starpu_profiling_start();

	STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
	initialized = INITIALIZED;
	/* Tell everybody that we initialized */
	STARPU_PTHREAD_COND_BROADCAST(&init_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);

	_STARPU_DEBUG("Initialisation finished\n");

#ifdef STARPU_USE_MP
	/* Finally, if we are a MP sink, we never leave this function. Else,
	 * we enter an infinite event loop which listen for MP commands from
	 * the source. */
	if (is_a_sink)
	{
		_starpu_sink_common_worker();

		/* We should normally never leave the loop as we don't want to
		 * really initialize STARPU */
		STARPU_ASSERT(0);
	}
#endif

	_starpu_catch_signals();

	/* if MPI is enabled, binding display will be done later, after MPI initialization */
	if (!_starpu_config.conf.will_use_mpi && starpu_getenv_number_default("STARPU_DISPLAY_BINDINGS", 0))
	{
		fprintf(stdout, "== Binding ==\n");
		starpu_display_bindings();
		fprintf(stdout, "== End of binding ==\n");
		fflush(stdout);
	}

	return 0;
}

/*
 * Handle runtime termination
 */

static void _starpu_terminate_workers(struct _starpu_machine_config *pconfig)
{
	int status = 0;
	unsigned workerid;
	unsigned n;

	starpu_wake_all_blocked_workers();

	for (workerid = 0; workerid < pconfig->topology.nworkers; workerid++)
	{
		_STARPU_DEBUG("wait for worker %u\n", workerid);

		struct _starpu_worker_set *set = pconfig->workers[workerid].set;
		struct _starpu_worker *worker = &pconfig->workers[workerid];

		/* in case StarPU termination code is called from a callback,
		 * we have to check if starpu_pthread_self() is the worker itself */
		if (set && set->nworkers > 0)
		{
			if (set->started)
			{
				if (!starpu_pthread_equal(starpu_pthread_self(), set->worker_thread))
					status = starpu_pthread_join(set->worker_thread, NULL);
				if (status)
				{
#ifdef STARPU_VERBOSE
					_STARPU_DEBUG("starpu_pthread_join -> %d\n", status);
#endif
				}
				set->started = 0;
			}
		}
		else
		{
			if (!worker->run_by_starpu)
				goto out;

			if (!starpu_pthread_equal(starpu_pthread_self(), worker->worker_thread))
				status = starpu_pthread_join(worker->worker_thread, NULL);
			if (status)
			{
#ifdef STARPU_VERBOSE
				_STARPU_DEBUG("starpu_pthread_join -> %d\n", status);
#endif
			}
		}

out:
		STARPU_ASSERT(starpu_task_prio_list_empty(&worker->local_tasks));
		for (n = 0; n < worker->local_ordered_tasks_size; n++)
			STARPU_ASSERT(worker->local_ordered_tasks[n] == NULL);
		_starpu_sched_ctx_list_delete(&worker->sched_ctx_list);
		free(worker->local_ordered_tasks);
		STARPU_ASSERT(_starpu_ctx_change_list_empty(&worker->ctx_change_list));
	}

	_starpu_trace_finalize();
}

/* Condition variable and mutex used to pause/resume. */
static starpu_pthread_cond_t pause_cond = STARPU_PTHREAD_COND_INITIALIZER;
static starpu_pthread_mutex_t pause_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;

void _starpu_may_pause(void)
{
	/* pause_depth is just protected by a memory barrier */
	STARPU_RMB();

	if (STARPU_UNLIKELY(_starpu_config.pause_depth > 0))
	{
		STARPU_PTHREAD_MUTEX_LOCK(&pause_mutex);
		while (_starpu_config.pause_depth > 0)
		{
			STARPU_PTHREAD_COND_WAIT(&pause_cond, &pause_mutex);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&pause_mutex);
	}
}

void starpu_pause()
{
	STARPU_HG_DISABLE_CHECKING(_starpu_config.pause_depth);
	_starpu_config.pause_depth += 1;

	starpu_fxt_trace_user_event_string("starpu_pause");
}

void starpu_resume()
{
	STARPU_PTHREAD_MUTEX_LOCK(&pause_mutex);
	_starpu_config.pause_depth -= 1;
	if (!_starpu_config.pause_depth)
	{
		STARPU_PTHREAD_COND_BROADCAST(&pause_cond);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&pause_mutex);

	starpu_fxt_trace_user_event_string("starpu_resume");
}

int starpu_is_paused()
{
	STARPU_RMB();
	return _starpu_config.pause_depth > 0;
}

unsigned _starpu_worker_can_block(unsigned memnode STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_NON_BLOCKING_DRIVERS
	return 0;
#else
	/* do not block if a sched_ctx change operation is pending */
	if (worker->state_changing_ctx_notice)
		return 0;

	unsigned can_block = 1;

	struct starpu_driver driver;
	driver.type = worker->arch;
	switch (driver.type)
	{
	case STARPU_CPU_WORKER:
		driver.id.cpu_id = worker->devid;
		break;
	case STARPU_CUDA_WORKER:
		driver.id.cuda_id = worker->devid;
		break;
	case STARPU_HIP_WORKER:
		driver.id.hip_id = worker->devid;
		break;
#ifdef STARPU_USE_OPENCL
	case STARPU_OPENCL_WORKER:
		starpu_opencl_get_device(worker->devid, &driver.id.opencl_id);
		break;
#endif
	default:
		goto always_launch;
	}
	if (!_starpu_may_launch_driver(&_starpu_config.conf, &driver))
		return 0;

always_launch:

#ifndef STARPU_SIMGRID
	if (!_starpu_check_that_no_data_request_exists(memnode))
		can_block = 0;
#endif

	if (!_starpu_machine_is_running())
		can_block = 0;

	if (!_starpu_execute_registered_progression_hooks())
		can_block = 0;

	return can_block;
#endif
}

static void _starpu_kill_all_workers(struct _starpu_machine_config *pconfig)
{
	/* set the flag which will tell workers to stop */
	ANNOTATE_HAPPENS_AFTER(&_starpu_config.running);
	pconfig->running = 0;
	/* running is just protected by a memory barrier */
	ANNOTATE_HAPPENS_BEFORE(&_starpu_config.running);
	STARPU_WMB();
	starpu_wake_all_blocked_workers();
}

void starpu_display_stats()
{
	starpu_profiling_bus_helper_display_summary();
	starpu_profiling_worker_helper_display_summary();
}

void starpu_shutdown(void)
{
	unsigned worker;
	STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
	init_count--;
	STARPU_ASSERT_MSG(init_count >= 0, "Number of calls to starpu_shutdown() can not be higher than the number of calls to starpu_init()\n");
	if (init_count)
	{
		_STARPU_DEBUG("Still somebody needing StarPU, don't deinitialize\n");
		STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);
		return;
	}

	/* We're last */
	initialized = CHANGING;
	STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);

	/* If the workers are frozen, no progress can be made. */
	STARPU_ASSERT_MSG(_starpu_config.pause_depth <= 0, "Did you forget to call starpu_resume before starpu_shutdown?");

	starpu_task_wait_for_no_ready();

	starpu_worker_wait_for_initialisation();

	/* tell all workers to shutdown */
	_starpu_kill_all_workers(&_starpu_config);

	unsigned i;
	unsigned nb_numa_nodes = starpu_memory_nodes_get_numa_count();
	for (i=0; i<nb_numa_nodes; i++)
	{
		_starpu_free_all_automatically_allocated_buffers(i);
	}

	{
	     int stats = starpu_getenv_number("STARPU_STATS");
	     if (stats != 0)
	     {
		  _starpu_display_msi_stats(stderr);
		  _starpu_display_alloc_cache_stats(stderr);
	     }
	}

	starpu_profiling_bus_helper_display_summary();
	starpu_profiling_worker_helper_display_summary();
	starpu_bound_clear();

	_starpu_deinitialize_registered_performance_models();

	_starpu_watchdog_shutdown();

	/* wait for their termination */
	_starpu_terminate_workers(&_starpu_config);

	{
	     int stats = starpu_getenv_number("STARPU_MEMORY_STATS");
	     if (stats != 0)
	     {
		  // Display statistics on data which have not been unregistered
		  starpu_data_display_memory_stats();
	     }
	}

	_starpu_delete_all_sched_ctxs();
	_starpu_sched_component_workers_destroy();

	for (worker = 0; worker < _starpu_config.topology.nworkers; worker++)
		_starpu_worker_deinit(&_starpu_config.workers[worker]);

	_starpu_profiling_terminate();

	_starpu_disk_unregister();
#ifdef STARPU_HAVE_HWLOC
	starpu_tree_free(_starpu_config.topology.tree);
	free(_starpu_config.topology.tree);
#endif
	_starpu_destroy_topology(&_starpu_config);
	_starpu_initialized_combined_workers = 0;

	_starpu_trace_finalize();

	_starpu_data_interface_shutdown();

	_starpu_job_fini();

	/* Drop all remaining tags */
	starpu_tag_clear();

#ifdef STARPU_OPENMP
	_starpu_omp_dummy_shutdown();
#endif
	_starpu_perf_knob_exit();
	_starpu_perf_counter_exit();
	_starpu_close_debug_logfile();

	_starpu_keys_initialized = 0;
	STARPU_PTHREAD_KEY_DELETE(_starpu_worker_key);
	STARPU_PTHREAD_KEY_DELETE(_starpu_worker_set_key);

	_starpu_task_deinit();

	STARPU_PTHREAD_MUTEX_LOCK(&init_mutex);
	initialized = UNINITIALIZED;
	/* Let someone else that wants to initialize it again do it */
	STARPU_PTHREAD_COND_SIGNAL(&init_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&init_mutex);

	/* Clear memory */
	free((char*) _starpu_config.conf.sched_policy_name);
	if (_starpu_config.conf.n_cuda_opengl_interoperability)
		free(_starpu_config.conf.cuda_opengl_interoperability);
	if (_starpu_config.conf.n_not_launched_drivers)
		free(_starpu_config.conf.not_launched_drivers);
	STARPU_AYU_FINISH();

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	if (_starpu_mpi_common_is_mp_initialized())
		_starpu_mpi_common_mp_deinit();
#endif

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
	if (_starpu_tcpip_common_is_mp_initialized())
		_starpu_tcpip_common_mp_deinit();
#endif
#ifdef STARPU_USE_MP
	_starpu_src_common_deinit();
#endif
	_starpu_print_idle_time();
	_STARPU_DEBUG("Shutdown finished\n");

#ifdef STARPU_NOSV
	{
		int status = nosv_shutdown();
		STARPU_ASSERT(status == 0);
		_STARPU_DISP("nOS-V: nosv_shutdown called\n");
	}
#endif
#ifdef STARPU_SIMGRID
	/* This finalizes the simgrid thread library, thus needs to be late */
	_starpu_simgrid_deinit();
#endif
}

unsigned starpu_worker_archtype_is_valid(enum starpu_worker_archtype type)
{
	return (type >= 0 && type < STARPU_NARCH)
		&& (starpu_driver_info[type].name_upper != NULL);
}

enum starpu_worker_archtype starpu_arch_mask_to_worker_archtype(unsigned mask)
{
	STARPU_ASSERT(mask && !(mask & (mask-1))); // ensures that only one bit of the mask is set

	enum starpu_worker_archtype worker_type = ffs(mask)-2; // ffs(mask) is the indice of the lesser bit

	STARPU_ASSERT(worker_type < STARPU_NARCH); // worker_type is positive and lesser than arch number
	STARPU_ASSERT(starpu_worker_archtype_is_valid(worker_type)); // worker_type is a valid worker architecture

	return worker_type;
}

#undef starpu_worker_get_count
unsigned starpu_worker_get_count(void)
{
	return _starpu_config.topology.nworkers;
}

void starpu_worker_get_current_task_exp_end(unsigned workerid, struct timespec *date)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	*date = worker->cl_expend;
}

unsigned starpu_worker_is_blocked_in_parallel(int workerid)
{
	if (!_starpu_worker_parallel_blocks)
		return 0;
	int relax_own_observation_state = 0;
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
	struct _starpu_worker *cur_worker = NULL;
	int cur_workerid = starpu_worker_get_id();
	if (workerid != cur_workerid)
	{
		/* in order to observe the 'blocked' state of a worker from
		 * another worker, we must avoid race conditions between
		 * 'blocked' state changes and state observations. This is the
		 * purpose of this 'if' block. */
		cur_worker = cur_workerid >= 0 ? _starpu_get_worker_struct(cur_workerid) : NULL;

		relax_own_observation_state = (cur_worker != NULL) && (cur_worker->state_relax_refcnt == 0);
		if (relax_own_observation_state && !worker->state_relax_refcnt)
		{
			/* moreover, when a worker (cur_worker != NULL)
			 * observes another worker, we need to take special
			 * care to avoid live locks, thus the observing worker
			 * must enter the relaxed state (if not relaxed
			 * already) before doing the observation in mutual
			 * exclusion */
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);

			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&cur_worker->sched_mutex);
			cur_worker->state_relax_refcnt = 1;
			STARPU_PTHREAD_COND_BROADCAST(&cur_worker->sched_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&cur_worker->sched_mutex);

			STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
		}
		/* the observer waits for a safe window to observe the state,
		 * and also waits for any pending blocking state change
		 * requests to be processed, in order to not obtain an
		 * ephemeral information */
		while (!worker->state_relax_refcnt
				|| worker->state_block_in_parallel_req
				|| worker->state_unblock_in_parallel_req)
		{
			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
		}
	}
	unsigned ret = _starpu_config.workers[workerid].state_blocked_in_parallel;
	/* once a worker state has been observed, the worker is 'tainted' for the next one full sched_op,
	 * to avoid changing the observed worker state - on which the observer
	 * made a scheduling decision - after the fact. */
	worker->state_blocked_in_parallel_observed = 1;
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	if (relax_own_observation_state)
	{
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&cur_worker->sched_mutex);
		cur_worker->state_relax_refcnt = 0;
		STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&cur_worker->sched_mutex);
	}
	return ret;
}

unsigned starpu_worker_is_slave_somewhere(int workerid)
{
	starpu_worker_lock(workerid);
	unsigned ret = _starpu_config.workers[workerid].is_slave_somewhere;
	starpu_worker_unlock(workerid);
	return ret;
}

int starpu_worker_get_count_by_type(enum starpu_worker_archtype type)
{
	unsigned n = 0;

	if (type != STARPU_ANY_WORKER)
	{
		if (type >= STARPU_NARCH)
			return -EINVAL;

		unsigned i;
		for (i = 0; i < _starpu_config.topology.ndevices[type]; i++)
			n += _starpu_config.topology.nworker[type][i];
		return n;
	}

	for (type = 0; type < STARPU_NARCH; type++)
		n += starpu_worker_get_count_by_type(type);
	return n;
}

unsigned starpu_combined_worker_get_count(void)
{
	return _starpu_config.topology.ncombinedworkers;
}

unsigned starpu_cpu_worker_get_count(void)
{
	return starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
}

unsigned starpu_cuda_worker_get_count(void)
{
	return starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
}


unsigned starpu_hip_worker_get_count(void)
{
	return starpu_worker_get_count_by_type(STARPU_HIP_WORKER);
}

unsigned starpu_opencl_worker_get_count(void)
{
	return starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER);
}

int starpu_asynchronous_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_copy;
}

int starpu_asynchronous_cuda_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_cuda_copy;
}


int starpu_asynchronous_hip_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_hip_copy;
}

int starpu_asynchronous_opencl_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_opencl_copy;
}

int starpu_asynchronous_max_fpga_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_max_fpga_copy;
}

int starpu_asynchronous_mpi_ms_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_mpi_ms_copy;
}

int starpu_asynchronous_tcpip_ms_copy_disabled(void)
{
	return _starpu_config.conf.disable_asynchronous_tcpip_ms_copy;
}

/* Return whether memory mapping is disabled (0) or enabled (1) */
int starpu_map_enabled(void)
{
	return _starpu_config.conf.enable_map;
}

int starpu_asynchronous_copy_disabled_for(enum starpu_node_kind kind)
{
	return asynchronous_copy_disabled[kind];
}

unsigned starpu_mpi_ms_worker_get_count(void)
{
	return starpu_worker_get_count_by_type(STARPU_MPI_MS_WORKER);
}

unsigned starpu_tcpip_ms_worker_get_count(void)
{
	return starpu_worker_get_count_by_type(STARPU_TCPIP_MS_WORKER);
}

/* When analyzing performance, it is useful to see what is the processing unit
 * that actually performed the task. This function returns the id of the
 * processing unit actually executing it, therefore it makes no sense to use it
 * within the callbacks of SPU functions for instance. If called by some thread
 * that is not controlled by StarPU, starpu_worker_get_id returns -1. */
#undef starpu_worker_get_id
int starpu_worker_get_id(void)
{
	struct _starpu_worker * worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->workerid;
	}
	else
	{
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}
#define starpu_worker_get_id _starpu_worker_get_id

#undef _starpu_worker_get_id_check
unsigned _starpu_worker_get_id_check(const char *f, int l)
{
	(void) f;
	(void) l;
	int id = _starpu_worker_get_id();
	STARPU_ASSERT_MSG(id>=0, "%s:%d Cannot be called from outside a worker\n", f, l);
	return id;
}

int starpu_combined_worker_get_id(void)
{
	struct _starpu_worker *worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->combined_workerid;
	}
	else
	{
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}

int starpu_combined_worker_get_size(void)
{
	struct _starpu_worker *worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->worker_size;
	}
	else
	{
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}

int starpu_combined_worker_get_rank(void)
{
	struct _starpu_worker *worker;

	worker = _starpu_get_local_worker_key();
	if (worker)
	{
		return worker->current_rank;
	}
	else
	{
		/* there is no worker associated to that thread, perhaps it is
		 * a thread from the application or this is some SPU worker */
		return -1;
	}
}

int starpu_worker_get_subworkerid(int id)
{
	return _starpu_config.workers[id].subworkerid;
}

int starpu_worker_get_devid(int id)
{
	return _starpu_config.workers[id].devid;
}

int starpu_worker_get_devnum(int id)
{
	return _starpu_config.workers[id].devnum;
}

unsigned starpu_worker_is_combined_worker(int id)
{
	return id >= (int)_starpu_config.topology.nworkers;
}

struct _starpu_combined_worker *_starpu_get_combined_worker_struct(unsigned id)
{
	unsigned basic_worker_count = starpu_worker_get_count();

	//_STARPU_DEBUG("basic_worker_count:%d\n",basic_worker_count);

	STARPU_ASSERT(id >= basic_worker_count);
	return &_starpu_config.combined_workers[id - basic_worker_count];
}

enum starpu_worker_archtype starpu_worker_get_type(int id)
{
	enum starpu_worker_archtype type = _starpu_config.workers[id].arch;
	STARPU_ASSERT(type < STARPU_NARCH);
	return type;
}

unsigned starpu_worker_get_ids_by_type(enum starpu_worker_archtype type, int *workerids, unsigned maxsize)
{
	unsigned nworkers = starpu_worker_get_count();

	unsigned cnt = 0;

	unsigned id;
	for (id = 0; id < nworkers; id++)
	{
		if (type == STARPU_ANY_WORKER || starpu_worker_get_type(id) == type)
		{
			/* Perhaps the array is too small ? */
			if (cnt >= maxsize)
				return -ERANGE;

			workerids[cnt++] = id;
		}
	}

	return cnt;
}

int starpu_worker_get_by_type(enum starpu_worker_archtype type, int num)
{
	unsigned nworkers = starpu_worker_get_count();

	int cnt = 0;

	unsigned id;
	for (id = 0; id < nworkers; id++)
	{
		if (type == STARPU_ANY_WORKER || starpu_worker_get_type(id) == type)
		{
			if (num == cnt)
				return id;
			cnt++;
		}
	}

	/* Not found */
	return -1;
}

int starpu_worker_get_by_devid(enum starpu_worker_archtype type, int devid)
{
	unsigned nworkers = starpu_worker_get_count();

	unsigned id;
	for (id = 0; id < nworkers; id++)
		if (starpu_worker_get_type(id) == type && starpu_worker_get_devid(id) == devid)
			return id;

	/* Not found */
	return -1;
}

int starpu_worker_get_devids(enum starpu_worker_archtype type, int *devids, int num)
{
	unsigned nworkers = starpu_worker_get_count();
	int workerids[nworkers];

	unsigned ndevice_workers = starpu_worker_get_ids_by_type(type, workerids, nworkers);

	unsigned ndevids = 0;

	if(ndevice_workers > 0)
	{
		unsigned id, devid;
		int cnt = 0;
		unsigned found = 0;
		for(id = 0; id < ndevice_workers; id++)
		{
			int curr_devid;
			curr_devid = _starpu_config.workers[workerids[id]].devid;
			for(devid = 0; devid < ndevids; devid++)
			{
				if(curr_devid == devids[devid])
				{
					found = 1;
					break;
				}
			}
			if(!found)
			{
				devids[ndevids++] = curr_devid;
				cnt++;
			}
			else
				found = 0;

			if(cnt == num)
				break;
		}
	}

	return ndevids;
}

unsigned starpu_worker_type_can_execute_task(enum starpu_worker_archtype worker_type, const struct starpu_task *task)
{
	return (STARPU_WORKER_TO_MASK(worker_type) & task->where) != 0;
}

void starpu_worker_get_name(int id, char *dst, size_t maxlen)
{
	char *name = _starpu_config.workers[id].name;

	snprintf(dst, maxlen, "%s", name);
}

int starpu_worker_get_bindid(int workerid)
{
	return _starpu_config.workers[workerid].bindid;
}

int starpu_bindid_get_workerids(int bindid, int **workerids)
{
	if (bindid >= (int) _starpu_config.nbindid)
		return 0;
	*workerids = _starpu_config.bindid_workers[bindid].workerids;
	return _starpu_config.bindid_workers[bindid].nworkers;
}

int starpu_worker_get_stream_workerids(unsigned devid, int *workerids, enum starpu_worker_archtype type)
{
	unsigned nworkers = starpu_worker_get_count();
	int nw = 0;
	unsigned id;
	for (id = 0; id < nworkers; id++)
	{
		if (_starpu_config.workers[id].devid == devid &&
		    (type == STARPU_ANY_WORKER || _starpu_config.workers[id].arch == type))
			workerids[nw++] = id;
	}
	return nw;
}

void starpu_worker_get_sched_condition(int workerid, starpu_pthread_mutex_t **sched_mutex, starpu_pthread_cond_t **sched_cond)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	*sched_cond = &_starpu_config.workers[workerid].sched_cond;
	*sched_mutex = &_starpu_config.workers[workerid].sched_mutex;
}

/* returns 1 if the call results in initiating a transition of worker WORKERID
 * from sleeping state to awake
 * returns 0 if worker WORKERID is not sleeping or the wake-up transition
 * already has been initiated
 */
static int starpu_wakeup_worker_locked(int workerid, starpu_pthread_cond_t *sched_cond, starpu_pthread_mutex_t *mutex STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	starpu_pthread_queue_broadcast(&_starpu_simgrid_task_queue[workerid]);
#endif
	if (_starpu_config.workers[workerid].status & STATUS_SLEEPING)
	{
		int ret = 0;
		if (_starpu_config.workers[workerid].state_keep_awake != 1)
		{
			_starpu_config.workers[workerid].state_keep_awake = 1;
			ret = 1;
		}
		/* cond_broadcast is required over cond_signal since
		 * the condition is share for multiple purpose */
		STARPU_PTHREAD_COND_BROADCAST(sched_cond);
		return ret;
	}
	else if (_starpu_config.workers[workerid].status & STATUS_SCHEDULING)
	{
		_starpu_config.workers[workerid].state_keep_awake = 1;
		return 0;
	}
	return 0;
}

static int starpu_wakeup_worker_no_relax(int workerid, starpu_pthread_cond_t *sched_cond, starpu_pthread_mutex_t *sched_mutex)
{
	int success;
	STARPU_PTHREAD_MUTEX_LOCK_SCHED(sched_mutex);
	success = starpu_wakeup_worker_locked(workerid, sched_cond, sched_mutex);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(sched_mutex);
	return success;
}

int starpu_wake_worker_locked(int workerid)
{
	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	return starpu_wakeup_worker_locked(workerid, sched_cond, sched_mutex);
}

int starpu_wake_worker_no_relax(int workerid)
{
	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	return starpu_wakeup_worker_no_relax(workerid, sched_cond, sched_mutex);
}

int _starpu_worker_get_nids_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize)
{
	unsigned nworkers = starpu_worker_get_count();

	int cnt = 0;

	unsigned id;
	for (id = 0; id < nworkers; id++)
	{
		if (type == STARPU_ANY_WORKER || starpu_worker_get_type(id) == type)
		{
			/* Perhaps the array is too small ? */
			if (cnt >= maxsize)
				return cnt;

			workerids[cnt++] = id;
		}
	}

	return cnt;
}

int _starpu_worker_get_nids_ctx_free_by_type(enum starpu_worker_archtype type, int *workerids, int maxsize)
{
	unsigned nworkers = starpu_worker_get_count();
	int cnt = 0;
	unsigned id;

	for (id = 0; id < nworkers; id++)
	{
		if (type == STARPU_ANY_WORKER || starpu_worker_get_type(id) == type)
		{
			/* Perhaps the array is too small ? */
			if (cnt >= maxsize)
				return cnt;
			unsigned found = 0;
			int s;
			for(s = 1; s < STARPU_NMAX_SCHED_CTXS; s++)
			{
				if(_starpu_config.sched_ctxs[s].id != STARPU_NMAX_SCHED_CTXS)
				{
					struct starpu_worker_collection *workers = _starpu_config.sched_ctxs[s].workers;
					struct starpu_sched_ctx_iterator it;

					workers->init_iterator(workers, &it);
					while(workers->has_next(workers, &it))
					{
						unsigned worker = workers->get_next(workers, &it);
						if(worker == id)
						{
							found = 1;
							break;
						}
					}

					if(found)
						break;
				}
			}
			if(!found)
				workerids[cnt++] = id;
		}
	}

	return cnt;
}

void starpu_get_version(int *major, int *minor, int *release)
{
	*major = STARPU_MAJOR_VERSION;
	*minor = STARPU_MINOR_VERSION;
	*release = STARPU_RELEASE_VERSION;
}

unsigned starpu_worker_get_sched_ctx_list(int workerid, unsigned **sched_ctxs)
{
	unsigned s = 0;
	unsigned nsched_ctxs = _starpu_worker_get_nsched_ctxs(workerid);
	_STARPU_MALLOC(*sched_ctxs, nsched_ctxs*sizeof(unsigned));
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct _starpu_sched_ctx_elt *e = NULL;
	struct _starpu_sched_ctx_list_iterator list_it;

	_starpu_sched_ctx_list_iterator_init(worker->sched_ctx_list, &list_it);
	while (_starpu_sched_ctx_list_iterator_has_next(&list_it))
	{
		e = _starpu_sched_ctx_list_iterator_get_next(&list_it);
		(*sched_ctxs)[s++] = e->sched_ctx;
	}
	return nsched_ctxs;
}

const char *starpu_worker_get_type_as_string(enum starpu_worker_archtype type)
{
	STARPU_ASSERT(type < STARPU_NARCH);
	const char *ret = starpu_driver_info[type].name_upper;
	if (!ret)
		ret = "unknown";
	return ret;
}

enum starpu_worker_archtype starpu_worker_get_type_from_string(const char *name)
{
	enum starpu_worker_archtype type;
	for (type = 0; type < STARPU_NARCH; type++)
	{
		if (!strcmp(name, starpu_driver_info[type].name_upper))
			return type;
	}
	return STARPU_UNKNOWN_WORKER;
}

const char *starpu_worker_get_type_as_env_var(enum starpu_worker_archtype type)
{
	STARPU_ASSERT(type < STARPU_NARCH);
	const char *ret = starpu_driver_info[type].name_var;
	if (!ret)
		ret = "UNKNOWN";
	return ret;
}

void _starpu_worker_set_stream_ctx(unsigned workerid, struct _starpu_sched_ctx *sched_ctx)
{
	STARPU_ASSERT(workerid < starpu_worker_get_count());
	struct _starpu_worker *w = _starpu_get_worker_struct(workerid);
	w->stream_ctx = sched_ctx;
}

struct _starpu_sched_ctx* _starpu_worker_get_ctx_stream(unsigned stream_workerid)
{
	if (stream_workerid >= starpu_worker_get_count())
		return NULL;
	struct _starpu_worker *w = _starpu_get_worker_struct(stream_workerid);
	return w->stream_ctx;
}

unsigned starpu_worker_get_sched_ctx_id_stream(unsigned stream_workerid)
{
	if (stream_workerid >= starpu_worker_get_count())
		return STARPU_NMAX_SCHED_CTXS;
	struct _starpu_worker *w = _starpu_get_worker_struct(stream_workerid);
	return w->stream_ctx != NULL ? w->stream_ctx->id : STARPU_NMAX_SCHED_CTXS;
}

void starpu_worker_display_count(FILE *output, enum starpu_worker_archtype type)
{
	int nworkers = starpu_worker_get_count_by_type(type);
	if (nworkers <= 0)
	{
		fprintf(output, "No %s worker\n", starpu_worker_get_type_as_string(type));
	}
	else
	{
		int ids[nworkers];
		starpu_worker_get_ids_by_type(type, ids, nworkers);
		fprintf(output, "%d %s worker%s\n", nworkers, starpu_worker_get_type_as_string(type), nworkers==1?"":"s");
	}
}

void starpu_worker_display_names(FILE *output, enum starpu_worker_archtype type)
{
	int nworkers;
	if (!starpu_driver_info[type].name_upper)
		return;

	nworkers = starpu_worker_get_count_by_type(type);
	if (nworkers <= 0)
	{
		fprintf(output, "No %s worker\n", starpu_worker_get_type_as_string(type));
	}
	else
	{
		int i, ids[nworkers];
		starpu_worker_get_ids_by_type(type, ids, nworkers);
		fprintf(output, "%d %s worker%s:\n", nworkers, starpu_worker_get_type_as_string(type), nworkers==1?"":"s");
		for(i = 0; i < nworkers; i++)
		{
			char name[256];
			starpu_worker_get_name(ids[i], name, 256);
			fprintf(output, "\t%s\n", name);
		}
	}
}

void starpu_worker_display_all(FILE *output)
{
	enum starpu_worker_archtype type;
	for (type = 0; type < STARPU_NARCH; type++)
		starpu_worker_display_names(output, type);
}

void _starpu_worker_refuse_task(struct _starpu_worker *worker, struct starpu_task *task)
{
	if (worker->pipeline_length || worker->arch == STARPU_OPENCL_WORKER)
	{
		int j;
		for (j = 0; j < worker->ntasks; j++)
		{
			const int j_mod = (j+worker->first_task)%STARPU_MAX_PIPELINE;
			if (task == worker->current_tasks[j_mod])
			{
				worker->current_tasks[j_mod] = NULL;
				if (j == 0)
				{
					worker->first_task = (worker->first_task + 1) % STARPU_MAX_PIPELINE;
					worker->current_task = NULL;
					_starpu_set_current_task(NULL);
				}
				break;
			}
		}
		STARPU_ASSERT(j<worker->ntasks);
	}
	else
	{
		worker->current_task = NULL;
		_starpu_set_current_task(NULL);
	}
	worker->ntasks--;
	task->prefetched = 0;
	int res = _starpu_push_task_to_workers(task);
	STARPU_ASSERT_MSG(res == 0, "_starpu_push_task_to_workers() unexpectedly returned = %d\n", res);
}

int starpu_worker_sched_op_pending(void)
{
	return _starpu_worker_sched_op_pending();
}

#undef starpu_worker_relax_on
void starpu_worker_relax_on(void)
{
	_starpu_worker_relax_on();
}

#undef starpu_worker_relax_off
void starpu_worker_relax_off(void)
{
	_starpu_worker_relax_off();
}

#undef starpu_worker_get_relax_state
int starpu_worker_get_relax_state(void)
{
	return _starpu_worker_get_relax_state();
}

#undef starpu_worker_lock
void starpu_worker_lock(int workerid)
{
	_starpu_worker_lock(workerid);
}

#undef starpu_worker_trylock
int starpu_worker_trylock(int workerid)
{
	return _starpu_worker_trylock(workerid);
}

#undef starpu_worker_unlock
void starpu_worker_unlock(int workerid)
{
	_starpu_worker_unlock(workerid);
}

#undef starpu_worker_lock_self
void starpu_worker_lock_self(void)
{
	_starpu_worker_lock_self();
}

#undef starpu_worker_unlock_self
void starpu_worker_unlock_self(void)
{
	_starpu_worker_unlock_self();
}

#undef starpu_wake_worker_relax
int starpu_wake_worker_relax(int workerid)
{
	return _starpu_wake_worker_relax(workerid);
}

#ifdef STARPU_HAVE_HWLOC
hwloc_cpuset_t starpu_worker_get_hwloc_cpuset(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return hwloc_bitmap_dup(worker->hwloc_cpu_set);
}
hwloc_obj_t starpu_worker_get_hwloc_obj(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	return worker->hwloc_obj;
}
#endif

/* Light version of _starpu_wake_worker_relax, which, when possible,
 * speculatively sets keep_awake on the target worker without waiting that
 * worker to enter the relaxed state.
 */
int starpu_wake_worker_relax_light(int workerid)
{
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	STARPU_ASSERT(worker != NULL);
	int cur_workerid = starpu_worker_get_id();
	if (workerid != cur_workerid)
	{
		starpu_worker_relax_on();

		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
		while (!worker->state_relax_refcnt)
		{
			/* Attempt a fast path if the worker is not really asleep */
			if (_starpu_config.workers[workerid].status & STATUS_SCHEDULING)
			{
				_starpu_config.workers[workerid].state_keep_awake = 1;
				STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
				starpu_worker_relax_off();
				return 1;
			}

			STARPU_PTHREAD_COND_WAIT(&worker->sched_cond, &worker->sched_mutex);
		}
	}
	else
	{
		STARPU_PTHREAD_MUTEX_LOCK_SCHED(&worker->sched_mutex);
	}
	int ret = starpu_wake_worker_locked(workerid);
	STARPU_PTHREAD_MUTEX_UNLOCK_SCHED(&worker->sched_mutex);
	if (workerid != cur_workerid)
	{
		starpu_worker_relax_off();
	}
	return ret;
}

#ifdef STARPU_WORKER_CALLBACKS
void starpu_worker_set_going_to_sleep_callback(void (*callback)(unsigned workerid))
{
	STARPU_ASSERT(_starpu_config.conf.callback_worker_going_to_sleep);
	_starpu_config.conf.callback_worker_going_to_sleep = callback;
}

void starpu_worker_set_waking_up_callback(void (*callback)(unsigned workerid))
{
	STARPU_ASSERT(_starpu_config.conf.callback_worker_waking_up);
	_starpu_config.conf.callback_worker_waking_up = callback;
}
#endif

enum starpu_node_kind starpu_worker_get_memory_node_kind(enum starpu_worker_archtype type)
{
	STARPU_ASSERT(type < STARPU_NARCH);
	enum starpu_node_kind kind = starpu_driver_info[type].memory_kind;
	STARPU_ASSERT_MSG(kind != (enum starpu_node_kind) -1, "no memory for archtype %d", type);
	return kind;
}
