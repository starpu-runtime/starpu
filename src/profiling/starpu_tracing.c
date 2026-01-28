/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2023-2025  École de Technologie Supérieure (ETS, Montréal)
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
#include <profiling/callbacks/callbacks.h>
#include <core/workers.h>
#include <core/jobs.h>
#include <core/dependencies/tags.h>
#include <sched_policies/sched_component.h>
#include <datawizard/memory_nodes.h>

#include "starpu_tracing.h"

#ifdef STARPU_PROF_TASKSTUBS
void _create_timer(struct _starpu_job *job, void *func)
{
	/* j-> job_successors: list of all the completion groups that depend on the job */
	char *name;
	unsigned long tid = job->job_id;

	tasktimer_argument_value_t args[1];
	args[0].type = TASKTIMER_LONG_INTEGER_TYPE;
	args[0].l_value = tid;

	uint64_t *parents = NULL;
	uint64_t myguid = tid;

	if (NULL != job->task->name)
	{
		name = job->task->name;
	}
	else
	{
		asprintf(&name, "%s %p", "UNRESOLVED ADDR", func);
	}

	unsigned nb_parents = 0;
	unsigned n;
	for(n=0 ; n < job->job_successors.ndeps; n++)
	{
		if (!job->job_successors.done[n])
		{
			struct _starpu_cg *cg = job->job_successors.deps[n];
			unsigned m;
			for(m=0 ; m < cg->ndeps ; m++)
			{
				if (!cg->done[m])
				{
					nb_parents += 1;
				}
			}
		}
	}
	if (nb_parents)
	{
		unsigned k = 0;
		unsigned n;
		_STARPU_MALLOC(parents, nb_parents*sizeof(uint64_t));
		for(n=0 ; n < job->job_successors.ndeps; n++)
		{
			if (!job->job_successors.done[n])
			{
				struct _starpu_cg *cg = job->job_successors.deps[n];
				unsigned m;
				for(m=0 ; m < cg->ndeps ; m++)
				{
					if (!cg->done[m])
					{
						struct _starpu_job *xjob = cg->deps[m];
						parents[k] = xjob->job_id;
						k ++;
					}
				}
			}
		}
	}

	TASKTIMER_CREATE(func, name, myguid, parents, nb_parents, tt);
	job->ps_task_timer = tt;
}
#endif

extern struct _starpu_machine_config _starpu_config;

int _starpu_trace_initialize()
{
#ifdef STARPU_USE_FXT
	_starpu_fxt_init_profiling(_starpu_config.conf.trace_buffer_size);
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_init)
	{
		struct starpu_prof_tool_info pi;

		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_init, 0, 0, starpu_prof_tool_driver_cpu, -1, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_init(&pi, NULL, NULL);
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS
	TASKTIMER_INITIALIZE();
#endif
	return 0;
}

int _starpu_trace_finalize()
{
#ifdef STARPU_USE_FXT
	_starpu_stop_fxt_profiling();
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_terminate)
	{
		struct starpu_prof_tool_info pi;
		pi = _starpu_prof_tool_get_info_init(starpu_prof_tool_event_terminate, 0, starpu_prof_tool_driver_cpu, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_terminate(&pi, NULL, NULL);
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS
	TASKTIMER_FINALIZE();
#endif

	return 0;
}

int _starpu_trace_initialize_begin()
{
#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_init_begin)
	{
		struct starpu_prof_tool_info pi;
		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_init_begin, 0, 0, starpu_prof_tool_driver_cpu, -1, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_init_begin(&pi, NULL, NULL);
	}
#endif
	return 0;
}


/**
 * A new memory node is registered.
 * \p nodeid is the id of the new node.
 */
int _starpu_trace_new_mem_node(int nodeid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBE2(_STARPU_FUT_NEW_MEM_NODE, nodeid, _starpu_gettid());
#endif
	return 0;
}

/**
 * A new worker thread is registered.
 * \p is the bind id the driver bound to (logical index).
 */
int _starpu_trace_register_thread(int bindid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBE2(FUT_NEW_LWP_CODE, bindid, _starpu_gettid());
#endif
	return 0;
}

int _starpu_trace_worker_initialize()
{
	return 0;
}

int _starpu_trace_worker_finalize()
{
	return 0;
}

/**
 * A worker has started its shutdown process.
 */
int _starpu_trace_worker_deinit_start()
{
#ifdef STARPU_USE_FXT
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBE1(_STARPU_FUT_WORKER_DEINIT_START, _starpu_gettid());
#endif

	return 0;
}

/**
 * A worker has completed its shutdown process.
 * \p workerkind is the worker id shut down.
 */
int _starpu_trace_worker_deinit_end(unsigned workerid, enum starpu_worker_archtype workerkind)
{
#ifdef STARPU_USE_FXT
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBE2(_STARPU_FUT_WORKER_DEINIT_END, _STARPU_FUT_WORKER_KEY(workerkind), _starpu_gettid());
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit)
	{
		enum starpu_prof_tool_driver_type drivertype;
		switch(workerkind)
		{
		case STARPU_CPU_WORKER: drivertype = starpu_prof_tool_driver_cpu; break;
		case STARPU_CUDA_WORKER: drivertype = starpu_prof_tool_driver_gpu; break;
		case STARPU_OPENCL_WORKER: drivertype = starpu_prof_tool_driver_ocl; break;
		case STARPU_HIP_WORKER: drivertype = starpu_prof_tool_driver_hip; break;
		default: drivertype = starpu_prof_tool_driver_cpu; break;
		}

		struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_deinit, workerid, workerid, drivertype, -1, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_deinit(&pi, NULL, NULL);
	}
#endif
	return 0;
}

int _starpu_trace_start_executing(struct _starpu_job *j, struct starpu_task *worker_task, struct _starpu_worker *worker, void *func)
{
#ifdef STARPU_USE_FXT
	/**
	 * The execution of the job starts at the device driver level.
	 * \p job is the job instance.
	 */
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_EXECUTING, _starpu_gettid(), (j)->job_id);
#endif

#ifdef STARPU_PROF_TOOL
	starpu_prof_tool_cb_func callback = NULL;
	/* First, we need to find if the worker is a cpu, a gpu, etc. */
	switch(worker->arch)
	{
	case STARPU_CPU_WORKER:
		callback = starpu_prof_tool_callbacks.starpu_prof_tool_event_start_cpu_exec;
		break;
	case STARPU_CUDA_WORKER:
	case STARPU_HIP_WORKER:
	case STARPU_OPENCL_WORKER:
		callback = starpu_prof_tool_callbacks.starpu_prof_tool_event_start_gpu_exec;
		break;
	default:
		callback = NULL;
	}

	if(callback)
	{
		struct starpu_prof_tool_info pi;
		int devid = worker->devid;

		enum starpu_prof_tool_event event_type = starpu_prof_tool_event_start_cpu_exec;
		enum starpu_prof_tool_driver_type driver_type = starpu_prof_tool_driver_cpu;
		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			event_type = starpu_prof_tool_event_start_cpu_exec;
			driver_type = starpu_prof_tool_driver_cpu;
			break;
		case STARPU_CUDA_WORKER:
		case STARPU_HIP_WORKER:
		case STARPU_OPENCL_WORKER:
			event_type = starpu_prof_tool_event_start_gpu_exec;
			driver_type = starpu_prof_tool_driver_gpu;
			break;
		default:
			goto out;
		}

		pi = _starpu_prof_tool_get_info(event_type, devid, worker_task->workerid, driver_type, -1, func);
		pi.model_name = _starpu_job_get_model_name(j);
		pi.task_name = _starpu_job_get_task_name(j);

		callback(&pi, NULL, NULL);
	out:
		;
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS

	unsigned long tid = j->job_id;
	char *name = NULL;
	/* a timer should have been created when the task was submitted */
	if(NULL == j->ps_task_timer)
	{
		_create_timer(j, func);
	}

	tasktimer_execution_space_t resource;
	resource.type = TASKTIMER_DEVICE_CPU;/* tmp until I find what to put here */
	resource.device_id = 0;
	resource.instance_id = _starpu_gettid;

	TASKTIMER_START(j->ps_task_timer, &resource);

//	if(NULL != name) free(name);
 #endif
	return 0;
}

/**
 * The execution of the job has been completed at the device driver level.
 * \p job is the job instance.
 */
int _starpu_trace_end_executing(struct _starpu_job *job, struct _starpu_worker *worker)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_EXECUTING, _starpu_gettid(), (job)->job_id);
#endif

#ifdef STARPU_PROF_TOOL
	starpu_prof_tool_cb_func callback = NULL;
	/* First, we need to find if the worker is a cpu, a gpu, etc. */
	switch(worker->arch)
	{
	case STARPU_CPU_WORKER:
		callback = starpu_prof_tool_callbacks.starpu_prof_tool_event_end_cpu_exec;
		break;
	case STARPU_CUDA_WORKER:
	case STARPU_HIP_WORKER:
	case STARPU_OPENCL_WORKER:
		callback = starpu_prof_tool_callbacks.starpu_prof_tool_event_end_gpu_exec;
		break;
	default:
		goto out;
	}

	if(callback)
	{
		struct starpu_task *worker_task = job->task;
		struct starpu_codelet *cl = worker_task->cl;
		// crash here
		void *func = _starpu_task_get_cpu_nth_implementation(cl, job->nimpl);

		struct starpu_prof_tool_info pi;
//	int devid = cpu_args->devid;
		// how do I get this? In the driver it is cpu_args->devid
		int devid = -1;
		enum starpu_prof_tool_driver_type driver_type;
		enum starpu_prof_tool_event event_type;

		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			event_type = starpu_prof_tool_event_end_cpu_exec;
			driver_type = starpu_prof_tool_driver_cpu;
			break;
		case STARPU_CUDA_WORKER:
		case STARPU_HIP_WORKER:
		case STARPU_OPENCL_WORKER:
			event_type = starpu_prof_tool_event_end_gpu_exec;
			driver_type = starpu_prof_tool_driver_gpu;
			break;
		default:
			goto out;
		}

		pi = _starpu_prof_tool_get_info(event_type, devid, worker->workerid, driver_type, -1, func);
		pi.model_name = _starpu_job_get_model_name(job);
		pi.task_name = _starpu_job_get_task_name(job);

		callback(&pi, NULL, NULL);
	}
 out:
	;
#endif

#ifdef STARPU_PROF_TASKSTUBS
	TASKTIMER_STOP(job->ps_task_timer);
	TASKTIMER_DESTROY(job->ps_task_timer);
#endif

	return 0;
}

/**
 * The execution of a codelet implementation routine of a task instance has been started.
 * \p job is the job instance.
 * \p nimpl is the routine implementation number in the codelet routines list for the worker architecture.
 * \p perf_arch is the performance model structure for the codelet on the worker architecture.
 * \p workerid is the id of the worker.
 * \p rank is the instance rank in a parallel team of workers in the case of a parallel task, or 0 for a sequential task.
 */
int _starpu_trace_start_codelet_body(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED, int nimpl STARPU_ATTRIBUTE_UNUSED, struct starpu_perfmodel_arch *perf_arch STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, int rank STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK|_STARPU_FUT_KEYMASK_TASK_VERBOSE|_STARPU_FUT_KEYMASK_DATA|_STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA) & fut_active))
	{
		int mem_node = workerid == -1 ? -1 : (int)starpu_worker_get_memory_node(workerid);
		int codelet_null = job->task->cl == NULL;
		int nowhere = (job->task->where == STARPU_NOWHERE) || (job->task->cl != NULL && job->task->cl->where == STARPU_NOWHERE);
		enum starpu_node_kind kind = workerid == -1 ? STARPU_UNUSED : starpu_worker_get_memory_node_kind(starpu_worker_get_type(workerid));
		FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_START_CODELET_BODY, job->job_id, (job->task)->sched_ctx, workerid, mem_node, _starpu_gettid(), (codelet_null == 1 || nowhere == 1));
		if (rank == 0 && job->task->cl && !nowhere)
		{
			const int __nbuffers = STARPU_TASK_GET_NBUFFERS(job->task);
			char __buf[FXT_MAX_PARAMS*sizeof(long)];
			int __i;
			for (__i = 0; __i < __nbuffers; __i++)
			{
				starpu_data_handle_t __handle = STARPU_TASK_GET_HANDLE(job->task, __i);
				void *__interface = _STARPU_TASK_GET_INTERFACES(job->task)[__i];
				if (__interface && __handle->ops->describe)
				{
					__handle->ops->describe(__interface, __buf, sizeof(__buf));
					_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_DATA, _STARPU_FUT_CODELET_DATA, workerid, _starpu_gettid(), __buf);
				}
				FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_CODELET_DATA_HANDLE, job->job_id, (__handle), _starpu_data_get_size(__handle), STARPU_TASK_GET_MODE(job->task, __i));
				/* Regarding the memory location:
				 * - if the data interface doesn't provide to_pointer operation, NULL will be returned
				 *   and the location will be -1, which is fine;
				 * - we have to check whether the memory is on an actual NUMA node (and not on GPU
				 *   memory, for instance);
				 * - looking at memory location before executing the task isn't the best choice:
				 *   the page can be not allocated yet. A solution would be to get the memory
				 *   location at the end of the task, but there is no FxT probe where we iterate over
				 *   handles, after task execution.
				 * */
				FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA, _STARPU_FUT_CODELET_DATA_HANDLE_NUMA_ACCESS, job->job_id, (__i), kind == STARPU_CPU_RAM && starpu_task_get_current_data_node(__i) >= 0 ? starpu_get_memory_location_bitmap(starpu_data_handle_to_pointer(__handle, (unsigned) starpu_task_get_current_data_node(__i)), starpu_data_get_size(__handle)) : -1);
			}
		}
		if (!(codelet_null == 1 || nowhere == 1))
		{
			const size_t __job_size = (perf_arch == NULL) ? 0 : _starpu_job_get_data_size(job->task->cl?job->task->cl->model:NULL, perf_arch, nimpl, job);
			const uint32_t __job_hash = (perf_arch == NULL) ? 0 : _starpu_compute_buffers_footprint(job->task->cl?job->task->cl->model:NULL, perf_arch, nimpl, job);
			FUT_FULL_PROBE8(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_CODELET_DETAILS, (job->task)->sched_ctx, __job_size, __job_hash, job->task->flops / 1000 / (job->task->cl && job->task->cl->type != STARPU_SEQ ? job->task_size : 1), job->task->tag_id, workerid, (job->job_id), _starpu_gettid());
		}
	}

#endif

	return 0;
}

/**
 * The execution of a codelet implementation routine of a task instance has been completed.
 * \p job is the job instance.
 * \p nimpl is the routine implementation number in the codelet routines list for the worker architecture.
 * \p perf_arch is the performance model structure for the codelet on the worker architecture.
 * \p workerid is the id of the worker.
 * \p rank is the instance rank in a parallel team of workers in the case of a parallel task, or 0 for a sequential task.
 */
int _starpu_trace_end_codelet_body(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED, unsigned nimpl STARPU_ATTRIBUTE_UNUSED, struct starpu_perfmodel_arch *perf_arch STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED, int rank STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active))
	{
		const size_t job_size = (perf_arch == NULL) ? 0 : _starpu_job_get_data_size(job->task->cl?job->task->cl->model:NULL, perf_arch, nimpl, job);
		const uint32_t job_hash = (perf_arch == NULL) ? 0 : _starpu_compute_buffers_footprint(job->task->cl?job->task->cl->model:NULL, perf_arch, nimpl, job);
		char _archname[32]="";
		if (perf_arch) starpu_perfmodel_get_arch_name(perf_arch, _archname, 32, 0);
		int nowhere = (job->task->where == STARPU_NOWHERE) || (job->task->cl != NULL && job->task->cl->where == STARPU_NOWHERE);
		int codelet_null = job->task->cl == NULL;
		_STARPU_FUT_FULL_PROBE6STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_END_CODELET_BODY, job->job_id, (job_size), (job_hash), workerid, _starpu_gettid(), (codelet_null == 1 || nowhere == 1), _archname);
	}
#endif

	return 0;
}

/**
 * A parallel team member of a parallel task has completed the task execution for its rank and enters the ending team synchronization barrier.
 * \p job is the job instance.
 */
int _starpu_trace_start_parallel_sync(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_PARALLEL_SYNC, _starpu_gettid(), (job)->job_id);
#endif
	return 0;
}

/**
 * A parallel team member of a parallel task has crossed the ending team synchronization barrier.
 * \p job is the job instance.
 */
int _starpu_trace_end_parallel_sync(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_PARALLEL_SYNC, _starpu_gettid(), (job)->job_id);
#endif
	return 0;
}

/**
 * The execution of a user callback associated to a task has been started.
 * \p job is the job instance.
 */
int _starpu_trace_start_callback(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_CALLBACK, job, _starpu_gettid());
#endif
	return 0;
}

/**
 * The execution of a user callback associated to a task has been completed.
 * \p job is the job instance.
 */
int _starpu_trace_end_callback(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_CALLBACK, job, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task is pushed to a worker.
 * \p task is the task instance.
 * \p prio is the priority of the task instance.
 */
int _starpu_trace_job_push(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int prio STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_JOB_PUSH, _starpu_get_job_associated_to_task(task)->job_id, prio, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task is poped from a queue.
 * \p task is the task instance.
 * \p prio is the priority of the task instance.
 */
int _starpu_trace_job_pop(struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int prio STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_JOB_POP, _starpu_get_job_associated_to_task(task)->job_id, prio, _starpu_gettid());
#endif
	return 0;
}

/**
 * Obsolete? Used only once with counter=0.
 */
int _starpu_trace_update_task_cnt(int counter STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_UPDATE_TASK_CNT, counter, _starpu_gettid());
#endif
	return 0;
}

/**
 * A synchronous data transfer has started to serve a task input dependence.
 * \p job is the job instance.
 */
int _starpu_trace_start_fetch_input(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_FETCH_INPUT_ON_TID, job, _starpu_gettid());
#endif
	return 0;
}

/**
 * A synchronous data transfer has completed serving a task input dependence.
 * \p job is the job instance.
 */
int _starpu_trace_end_fetch_input(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_FETCH_INPUT_ON_TID, job, _starpu_gettid());
#endif
	return 0;
}

/**
 * A data transfer has started to serve a task output dependence.
 * \p job is the job instance.
 */
int _starpu_trace_start_push_output(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_PUSH_OUTPUT_ON_TID, job, _starpu_gettid());
#endif
	return 0;
}

/**
 * A data transfer has completed serving a task output dependence.
 * \p job is the job instance.
 */
int _starpu_trace_end_push_output(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_PUSH_OUTPUT_ON_TID, job, _starpu_gettid());
#endif
	return 0;
}

/**
 * An asynchronous data transfer has completed serving a task input dependence.
 * \p job is the job instance.
 * \p id is the worker id.
 * Note: This trace event does not seem to be used.
 */
int _starpu_trace_worker_end_fetch_input(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED, int id STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_FETCH_INPUT, job, id);
#endif
	return 0;
}

/**
 * An asynchronous data transfer has started to serve a task input dependence.
 * \p job is the job instance. It is NULL in every occurrence.
 * \p id is the worker id.
 */
int _starpu_trace_worker_start_fetch_input(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED, int id STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_FETCH_INPUT, job, id);
#endif
	return 0;
}

/**
 * A task is associated with a dependence tag.
 * \p tag is the tag id.
 * \p job is the job instance.
 */
int _starpu_trace_tag(starpu_tag_t *tag STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG, tag, (job)->job_id);
#endif
	return 0;
}

/**
 * A dependence is declared between two tags.
 * \p tag_child is the successor dependence tag id.
 * \p tag_parent is the predecessor dependence tag id.
 */
int _starpu_trace_tag_deps(starpu_tag_t *tag_child STARPU_ATTRIBUTE_UNUSED, starpu_tag_t *tag_parent STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG_DEPS, tag_child, tag_parent);
#endif
	return 0;
}

/**
 * A dependence is declared between two tasks.
 * \p job_prev is the predecessor job.
 * \p job_succ is the successor job.
 */
int _starpu_trace_task_deps(struct _starpu_job *job_prev STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *job_succ STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_DEPS, (job_prev)->job_id, (job_succ)->job_id, (job_succ)->task->type, 1, "task");
#endif

#ifdef STARPU_PROF_TASKSTUBS
	/* looks like the succ is the current task */

#if 0
	if (job_succ)
		printf("succ: %d\t", job_succ->job_id);
	if (job_prev)
		printf("prev: %d", job_prev->job_id);
	printf("\n");
	if (!job_prev->ps_task_timer)
		printf("prev does not have any timer\n");
	if (!job_succ->ps_task_timer)
		printf("succ does not have any timer\n");
	if (!job_prev->ps_task_timer)
		_create_timer(job_prev, NULL);

	TASKTIMER_ADD_CHILDREN(job_prev->ps_task_timer, job_succ->job_id, 1);
//	TASKTIMER_ADD_PARENTS(job_succ->ps_task_timer, job_prev->job_id, 1);
#endif
#endif
	return 0;
}

/**
 * An end dependence between a predecessor task and successor task whose completion had been deferred has now been resolved.
 * \p job_prev is the predecessor job.
 * \p job_succ is the successor job with deferred completion.
 */
int _starpu_trace_task_end_dep(struct _starpu_job *job_prev STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *job_succ STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_DO_PROBE2(_STARPU_FUT_TASK_END_DEP, (job_prev)->job_id, (job_succ)->job_id);
#endif
	return 0;
}

/**
 * A dependence edge between a defunct task and a successor task has been detected.
 * \p ghost_prev_id is the predecessor ghost id.
 * \p job_succ is the successor job.
 */
int _starpu_trace_ghost_task_deps(unsigned ghost_prev_id STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *job_succ STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_DEPS, (ghost_prev_id), (job_succ)->job_id, (job_succ)->task->type, 1, "ghost");
#endif
	return 0;
}

int _starpu_trace_recursive_task_deps(unsigned long prev_id STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *job_succ STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_RECURSIVE_TASKS
#ifdef STARPU_USE_FXT
	_STARPU_FUT_FULL_PROBE4STR(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_DEPS, (prev_id), (job_succ)->job_id, (job_succ)->task->type, 1, "recursive_task");
#endif
#endif
	return 0;
}

int _starpu_trace_recursive_task(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_RECURSIVE_TASKS
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active))
	{
		unsigned int is_recursive_task=(job)->recursive.is_recursive_task;
		unsigned long recursive_task_parent=(job)->task->recursive_task_parent;
		FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_RECURSIVE_TASK, (job)->job_id, is_recursive_task, recursive_task_parent);
	}
#endif
#endif
	return 0;
}

/**
 * A task is marked to be ignored in debugging tools.
 * \p job is the job to be ignored.
 */
int _starpu_trace_task_exclude_from_dag(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	unsigned exclude_from_dag = (job)->exclude_from_dag;
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_EXCLUDE_FROM_DAG, (job)->job_id, (long unsigned)exclude_from_dag);
#endif
	return 0;
}

/**
 * A task is assigned a name, line and color metadata.
 * \p job is the corresponding job.
 */
int _starpu_trace_task_name_line_color(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_starpu_trace_task_color(job);
	_starpu_trace_task_name(job);
	_starpu_trace_task_line(job);
#endif
	return 0;
}

/**
 * A task is assigned a line metadata.
 * \p job is the corresponding job.
 */
int _starpu_trace_task_line(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if ((job)->task->file)
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_LINE, (job)->job_id, (job)->task->line, (job)->task->file);
#endif
	return 0;
}

/**
 * A task is assigned a name metadata.
 * \p job is the corresponding job.
 */
int _starpu_trace_task_name(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active))
	{
		const char *model_name = _starpu_job_get_model_name((job));
		const char *name = _starpu_job_get_task_name((job));
		if (name)
		{
			_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_NAME, (job)->job_id, _starpu_gettid(), name);
		}
		else
		{
			_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_NAME, (job)->job_id, _starpu_gettid(), "unknown");
		}
		if (model_name)
			_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_MODEL_NAME, (job)->job_id, _starpu_gettid(), model_name);
	}
#endif
	return 0;
}

/**
 * A task is assigned a color metadata.
 * \p job is the corresponding job.
 */
int _starpu_trace_task_color(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active))
	{
		if ((job)->task->color != 0)
			FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_COLOR, (job)->job_id, (job)->task->color);
		else if ((job)->task->cl && (job)->task->cl->color != 0)
			FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_COLOR, (job)->job_id, (job)->task->cl->color);
	}
#endif
	return 0;
}

/**
 * A task has completed its codelet routine execution and its epilogue steps.
 * \p job is the completed job.
 */
int _starpu_trace_task_done(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_DONE, (job)->job_id, _starpu_gettid());
#endif
	return 0;
}

/**
 * A dependence tag is releasing its dependences.
 * \p tag is the done tag.
 */
int _starpu_trace_tag_done(struct _starpu_tag *tag STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_TASK) & fut_active))
	{
		struct _starpu_job *job = (tag)->job;
		const char *model_name = _starpu_job_get_task_name((job));
		if (model_name)
		{
			_STARPU_FUT_FULL_PROBE3STR(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG_DONE, (tag)->id, _starpu_gettid(), 1, model_name);
		}
		else
		{
			FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TAG_DONE, (tag)->id, _starpu_gettid(), 0);
		}
	}
#endif
	return 0;
}

/**
 * A data handle is assigned a name metadata.
 * \p handle is the corresponding data handle.
 * \p name is the name to be assigned to the data handle.
 */
int _starpu_trace_data_name(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, const char *name STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_FUT_FULL_PROBE1STR(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_NAME, handle, name);
#endif
	return 0;
}

/**
 * A data handle is assigned coordinates metadata.
 * \p handle is the corresponding data handle.
 * \p dim is the number of dimensions for the coordinates.
 * \p v is the array of coordinates.
 */
int _starpu_trace_data_coordinates(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned dim STARPU_ATTRIBUTE_UNUSED, int v[] STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	switch (dim)
	{
	case 1: FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0]); break;
	case 2: FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1]); break;
	case 3: FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1], v[2]); break;
	case 4: FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1], v[2], v[3]); break;
	default: FUT_FULL_PROBE7(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_DATA_COORDINATES, handle, dim, v[0], v[1], v[2], v[3], v[4]); break;
	}
#endif
	return 0;
}

/**
 * A copy of data has been performed from one memory node to another memory node.
 * \p src_node is the source node of the copy.
 * \p dst_node is the destination node of the copy.
 * \p size is the length of the copy in bytes.
 */
int _starpu_trace_data_copy(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_DATA_COPY, src_node, dst_node, size);
#endif
	return 0;
}

/**
 * A data handle has been marked as eligible for cache eviction.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_data_wont_use(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DATA, _STARPU_FUT_DATA_WONT_USE, handle, _starpu_fxt_get_submit_order(), _starpu_fxt_get_job_id(), _starpu_gettid());
#endif
	return 0;
}

/**
 * A data handle cache eviction mark is being processed.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_data_doing_wont_use(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_DATA_DOING_WONT_USE, handle);
#endif
	return 0;
}

/**
 * A data copy has been started by a device driver between two nodes.
 * \p src_node is the source node.
 * \p dst_node is the destination node.
 * \p size is the data copy length in bytes.
 * \p com_id is the communication id.
 * \p prefetch is the prefetch level.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_start_driver_copy(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED, unsigned long com_id STARPU_ATTRIBUTE_UNUSED, enum starpu_is_prefetch prefetch STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_DRIVER_COPY, src_node, dst_node, size, com_id, prefetch, handle);
#endif
	return 0;
}

/**
 * A synchronous data copy has been completed by a device driver between two nodes.
 * \p src_node is the source node.
 * \p dst_node is the destination node.
 * \p size is the data copy length in bytes.
 * \p com_id is the communication id.
 * \p prefetch is the prefetch level.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_end_driver_copy(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED, unsigned long com_id STARPU_ATTRIBUTE_UNUSED,enum starpu_is_prefetch prefetch STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_DRIVER_COPY, src_node, dst_node, size, com_id, prefetch);
#endif
	return 0;
}

/**
 * An asynchronous data copy has been started by a device driver between two nodes.
 * \p src_node is the source node.
 * \p dst_node is the destination node.
 */
int _starpu_trace_start_driver_copy_async(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_DRIVER_COPY_ASYNC, src_node, dst_node);
#endif
	return 0;
}

/**
 * A synchronous data copy has been completed by a device driver between two nodes.
 * \p src_node is the source node.
 * \p dst_node is the destination node.
 */
int _starpu_trace_end_driver_copy_async(unsigned src_node STARPU_ATTRIBUTE_UNUSED, unsigned dst_node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_DRIVER_COPY_ASYNC, src_node, dst_node);
#endif
	return 0;
}

/**
 * A task has been stolen by an idle worker from a victim worker.
 * \p empty_q is the workerid of the idle thief worker.
 * \p victim_q is the workerid of the victim worker.
 */
int _starpu_trace_work_stealing(unsigned empty_q STARPU_ATTRIBUTE_UNUSED, unsigned victim_q STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_SCHED_VERBOSE, _STARPU_FUT_WORK_STEALING, empty_q, victim_q);
#endif
	return 0;
}

/**
 * A worker has started electing a new task to execute.
 */
int _starpu_trace_worker_scheduling_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A worker has completed electing a new task to execute.
 */
int _starpu_trace_worker_scheduling_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_END, _starpu_gettid());
#endif
	return 0;
}

int _starpu_trace_worker_scheduling_push()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_PUSH, _starpu_gettid());
#endif
	return 0;
}

int _starpu_trace_worker_scheduling_pop()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_WORKER_SCHEDULING_POP, _starpu_gettid());
#endif
	return 0;
}

/**
 * An idle worker has fell asleep.
 */
int _starpu_trace_worker_sleep_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER, _STARPU_FUT_WORKER_SLEEP_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * An idle worker has woken up.
 */
int _starpu_trace_worker_sleep_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_WORKER, _STARPU_FUT_WORKER_SLEEP_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A new task has been submitted.
 * \p job is the new job being submitted.
 * \p iter is the optional outermost iteration number metadata in which the task submission occurs.
 * \p iter is the optional innermost iteration number metadata in which the task submission occurs.
 */
int _starpu_trace_task_submit(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED, long iter STARPU_ATTRIBUTE_UNUSED, long subiter STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *pjob STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE7(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_SUBMIT, (job)->job_id, iter, subiter, (job)->task->no_submitorder?0:_starpu_fxt_get_submit_order(), (job)->task->priority, (job)->task->type, _starpu_gettid());
#ifdef STARPU_RECURSIVE_TASKS
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_RECURSIVE_SUBMIT, (job)->job_id, (job)->recursive.level, ((pjob) ? (pjob)->job_id : 0));
#endif
#endif
#ifdef STARPU_PROF_TASKSTUBS
// unsigned long starpu_task_get_job_id(struct starpu_task *task);

	if(NULL == job->ps_task_timer)
	{
		_create_timer(job, NULL);
	}

	unsigned n;
	for(n=0 ; n < job->job_successors.ndeps; n++)
	{
		if (!job->job_successors.done[n])
		{
			struct _starpu_cg *cg = job->job_successors.deps[n];
			unsigned m;
			for(m=0 ; m < cg->ndeps ; m++)
			{
				if (!cg->done[m])
				{
					struct _starpu_job *xjob = cg->deps[m];
					uint64_t parent[1];
					parent[0] = xjob->job_id;
					TASKTIMER_ADD_PARENTS(job->ps_task_timer, parent, 1);
				}
			}
		}
	}

#endif

	return 0;
}

/**
 * A task submission process has been started.
 */
int _starpu_trace_task_submit_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_SUBMIT_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task submission process has been completed.
 */
int _starpu_trace_task_submit_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_SUBMIT_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task submission throttling process has been engaged, task submission will be blocked until the throttling process gets disengaged.
 */
int _starpu_trace_task_throttle_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_THROTTLE_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task submission throttling process has been disengaged.
 */
int _starpu_trace_task_throttle_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_THROTTLE_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task building operation has been started.
 */
int _starpu_trace_task_build_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_BUILD_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A task building operation has been completed.
 */
int _starpu_trace_task_build_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_BUILD_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A StarPU-MPI task decoding operation has been started.
 */
int _starpu_trace_task_mpi_decode_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_DECODE_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A StarPU-MPI task decoding operation has been completed.
 */
int _starpu_trace_task_mpi_decode_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_DECODE_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A StarPU-MPI pre-task communication phase has been started.
 */
int _starpu_trace_task_mpi_pre_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_PRE_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A StarPU-MPI pre-task communication phase has been completed.
 */
int _starpu_trace_task_mpi_pre_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_PRE_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A StarPU-MPI post-task communication phase has been started.
 */
int _starpu_trace_task_mpi_post_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_POST_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A StarPU-MPI post-task communication phase has been completed.
 */
int _starpu_trace_task_mpi_post_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_MPI_VERBOSE, _STARPU_FUT_TASK_MPI_POST_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A wait operation on a specific task has been started.
 * \p job is the task being waited for completion.
 */
int _starpu_trace_task_wait_start(struct _starpu_job *job STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_START, (job)->job_id, _starpu_gettid());
#endif
	return 0;
}

/**
 * A wait operation on a specific task has been completed.
 */
int _starpu_trace_task_wait_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A wait operation on all submitted tasks has been started.
 */
int _starpu_trace_task_wait_for_all_start()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_FOR_ALL_START, _starpu_gettid());
#endif
	return 0;
}

/**
 * A wait operation on all submitted tasks has been completed.
 */
int _starpu_trace_task_wait_for_all_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_TASK_VERBOSE, _STARPU_FUT_TASK_WAIT_FOR_ALL_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A fresh memory allocation operation has been started.
 * \p memnode is the memory node on which the allocation is requested.
 * \p size is the size in bytes of the allocation request.
 * \p handle is the corresponding data handle.
 * \p is_prefetch is a boolean indicating whether the operation is speculative of performed by necessity.
 */
int _starpu_trace_start_alloc(unsigned memnode STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, enum starpu_is_prefetch is_prefetch STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_ALLOC, memnode, _starpu_gettid(), size, handle, is_prefetch);
#endif
	return 0;
}

/**
 * A fresh memory allocation operation has been completed.
 * \p memnode is the memory node on which the allocation is requested.
 * \p handle is the corresponding data handle.
 * \p r is the size of the memory allocated.
 */
int _starpu_trace_end_alloc(unsigned memnode STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, starpu_ssize_t r STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_ALLOC, memnode, _starpu_gettid(), handle, r);
#endif
	return 0;
}

/**
 * A fresh or cached memory allocation operation has been started.
 * \p memnode is the memory node on which the allocation is requested.
 * \p size is the size in bytes of the allocation request.
 * \p handle is the corresponding data handle.
 * \p is_prefetch is a boolean indicating whether the operation is speculative of performed by necessity.
 */
int _starpu_trace_start_alloc_reuse(unsigned memnode STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, enum starpu_is_prefetch is_prefetch STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_ALLOC_REUSE, memnode, _starpu_gettid(), size, handle, is_prefetch);
#endif
	return 0;
}

/**
 * A fresh or cached memory allocation operation has been completed.
 * \p memnode is the memory node on which the allocation is requested.
 * \p handle is the corresponding data handle.
 * \p r is the size of the memory allocated.
 */
int _starpu_trace_end_alloc_reuse(unsigned memnode STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, starpu_ssize_t r STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_ALLOC_REUSE, memnode, _starpu_gettid(), handle, r);
#endif
	return 0;
}

/**
 * A memory free operation has been started.
 * \p memnode is the memory node on which the allocation is requested.
 * \p size is the size of the memory allocated.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_start_free(unsigned memnode STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_FREE, memnode, _starpu_gettid(), size, handle);
#endif
	return 0;
}

/**
 * A memory free operation has been completed.
 * \p memnode is the memory node on which the allocation is requested.
 * \p size is the size of the memory allocated.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_end_free(unsigned memnode STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_FREE, memnode, _starpu_gettid(), handle);
#endif
	return 0;
}

/**
 * A synchronous cache writeback data transfer has been started.
 * \p memnode is the destination node.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_start_writeback(unsigned memnode STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_WRITEBACK, memnode, _starpu_gettid(), handle);
#endif
	return 0;
}

/**
 * A synchronous cache writeback data transfer has been completed.
 * \p memnode is the destination node.
 * \p handle is the corresponding data handle.
 */
int _starpu_trace_end_writeback(unsigned memnode STARPU_ATTRIBUTE_UNUSED, starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_WRITEBACK, memnode, _starpu_gettid(), handle);
#endif
	return 0;
}

/**
 * The memory usage statistics on a memory node has been updated.
 * \p memnode is the corresponding memory node.
 * \p used is the updated amount of memory used on the memory node, in bytes.
 */
int _starpu_trace_used_mem(unsigned memnode STARPU_ATTRIBUTE_UNUSED, size_t used STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_USED_MEM, memnode, used, _starpu_gettid());
#endif
	return 0;
}

/**
 * A memory reclaiming process has been started.
 * \p memnode is the destination node.
 * \p is_prefetch is a boolean indicating whether the operation is speculative of performed by necessity.
 */
int _starpu_trace_start_memreclaim(unsigned memnode STARPU_ATTRIBUTE_UNUSED,enum starpu_is_prefetch is_prefetch STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_MEMRECLAIM, memnode, is_prefetch, _starpu_gettid());
#endif
	return 0;
}

/**
 * A memory reclaiming process has been completed.
 * \p memnode is the destination node.
 * \p is_prefetch is a boolean indicating whether the operation is speculative of performed by necessity.
 */
int _starpu_trace_end_memreclaim(unsigned memnode STARPU_ATTRIBUTE_UNUSED, enum starpu_is_prefetch is_prefetch STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_MEMRECLAIM, memnode, is_prefetch, _starpu_gettid());
#endif
	return 0;
}

/**
 * An asynchronous cache writeback data transfer has been started.
 * \p memnode is the destination node.
 */
int _starpu_trace_start_writeback_async(unsigned memnode STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_START_WRITEBACK_ASYNC, memnode, _starpu_gettid());
#endif
	return 0;
}

/**
 * An asynchronous cache writeback data transfer has been completed.
 * \p memnode is the destination node.
 */
int _starpu_trace_end_writeback_async(unsigned memnode STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_END_WRITEBACK_ASYNC, memnode, _starpu_gettid());
#endif
	return 0;
}

/**
 * A PAPI task event has been collected.
 * \p event_id is the PAPI event id.
 * \p task is the corresponding task.
 * \p value is the value collected from the PAPI event.
 */
int _starpu_trace_papi_task_event(int event_id STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, long long int value STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_DO_PROBE3(_STARPU_FUT_PAPI_TASK_EVENT_VALUE, event_id, _starpu_get_job_associated_to_task(task)->job_id, value);
#endif
	return 0;
}

/* We skip these events because they are called so often that they cause FxT to
 * fail and make the overall trace unreadable anyway. */
/**
 * A data transfer progression phase has been started for a memory node.
 * \p memnode is the corresponding memory node.
 */
int _starpu_trace_start_progress(unsigned memnode STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_START_PROGRESS_ON_TID, memnode, _starpu_gettid());
#endif

#ifdef STARPU_PROF_TOOL
	if (worker && starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer)
	{
		struct starpu_prof_tool_info pi;
		enum starpu_prof_tool_driver_type driver_type;
		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			driver_type = starpu_prof_tool_driver_cpu;
			break;
		case STARPU_CUDA_WORKER:
		case STARPU_HIP_WORKER:
		case STARPU_OPENCL_WORKER:
			driver_type = starpu_prof_tool_driver_gpu;
			break;
		default:
			goto out;
		}

//	pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_start_transfer, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, cpu_worker->nb_buffers_totransfer, cpu_worker->nb_buffers_transferred);
		// we can pass more info here
		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, driver_type, memnode, 0, 0);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
	out:
		;
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS
	//TASKTIMER_DATA_TRANSFER_RESUME(100); /* TODO */
#endif

return 0;
}

/**
 * A data transfer progression phase has been completed for a memory node.
 * \p memnode is the corresponding memory node.
 */
int _starpu_trace_end_progress(unsigned memnode STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_WORKER_VERBOSE, _STARPU_FUT_END_PROGRESS_ON_TID, memnode, _starpu_gettid());
#endif

#ifdef STARPU_PROF_TOOL
	if(worker && starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer)
	{
		struct starpu_prof_tool_info pi;
		enum starpu_prof_tool_driver_type driver_type;
		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			driver_type = starpu_prof_tool_driver_cpu;
			break;
		case STARPU_CUDA_WORKER:
		case STARPU_HIP_WORKER:
		case STARPU_OPENCL_WORKER:
			driver_type = starpu_prof_tool_driver_gpu;
			break;
		default:
			goto out;
		}

		// we can pass more info here
		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, driver_type, memnode, 0, 0);

//	pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, cpu_worker->nb_buffers_totransfer, cpu_worker->nb_buffers_transferred);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
	out:
		;
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS
	//TASKTIMER_DATA_TRANSFER_STOP(100); /* TODO */
#endif

	return 0;
}

/**
 * A user-defined event has occurred.
 * \p code is the user-defined event code
 */
int _starpu_trace_user_event(unsigned long code STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_USER, _STARPU_FUT_USER_EVENT, code, _starpu_gettid());
#endif
	return 0;
}

/**
 * A trace meta-event has been recorded.
 * \p S is the trace meta-event string.
 */
int _starpu_trace_meta(const char *S STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
#ifdef FUT_DO_ALWAYS_PROBESTR
	FUT_FULL_PROBESTR(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_EVENT,S);
#endif
#endif
	return 0;
}

/**
 * The profiling status has been updated.
 * \p status is the new profiling status.
 */
int _starpu_trace_set_profiling(int status STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_SET_PROFILING, status, _starpu_gettid());
#endif
	return 0;
}

/**
 * Obsolete? Does not seem to be used anymore.
 */
int _starpu_trace_task_wait_for_all()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE0(_STARPU_FUT_KEYMASK_TASK, _STARPU_FUT_TASK_WAIT_FOR_ALL);
#endif
	return 0;
}

/**
 * An unconditional event has been recorded.
 * \p S is the event string.
 */
int _starpu_trace_event_always(const char *S STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
#ifdef FUT_DO_ALWAYS_PROBESTR
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBESTR(_STARPU_FUT_EVENT,S);
#endif
#endif
	return 0;
}

/**
 * An default verbosity level event has been recorded.
 * \p S is the event string.
 */
int _starpu_trace_event(const char *S STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
#ifdef FUT_DO_ALWAYS_PROBESTR
	FUT_FULL_PROBESTR(_STARPU_FUT_KEYMASK_EVENT, _STARPU_FUT_EVENT,S);
#endif
#endif
	return 0;
}

/**
 * An verbose level event has been recorded.
 * \p S is the event string.
 */
int _starpu_trace_event_verbose(const char *S STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
#ifdef FUT_DO_ALWAYS_PROBESTR
	FUT_FULL_PROBESTR(_STARPU_FUT_KEYMASK_EVENT_VERBOSE, _STARPU_FUT_EVENT,S);
#endif
#endif
	return 0;
}

/**
 * Obsolete? Does not seem to be used anymore.
 */
int _starpu_trace_thread_event(const char *S STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_FUT_FULL_PROBE1STR(_STARPU_FUT_KEYMASK_WORKER, _STARPU_FUT_THREAD_EVENT, _starpu_gettid(), S);
#endif
	return 0;
}

/**
 * A scheduling context hypervisor operation has been started.
 */
int _starpu_trace_hypervisor_begin()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_HYP, _STARPU_FUT_HYPERVISOR_BEGIN, _starpu_gettid());
#endif
	return 0;
}

/**
 * A scheduling context hypervisor operation has been completed.
 */
int _starpu_trace_hypervisor_end()
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_HYP, _STARPU_FUT_HYPERVISOR_END, _starpu_gettid());
#endif
	return 0;
}

/**
 * A mutex lock operation has been started.
 */
int _starpu_trace_locking_mutex()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_LOCKING_MUTEX,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A mutex lock operation has been completed.
 */
int _starpu_trace_mutex_locked()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_MUTEX_LOCKED,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A mutex unlock operation has been started.
 */
int _starpu_trace_unlocking_mutex()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_UNLOCKING_MUTEX,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A mutex unlock operation has been completed.
 */
int _starpu_trace_mutex_unlocked()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_MUTEX_UNLOCKED,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A mutex trylock operation has been attempted.
 */
int _starpu_trace_trylock_mutex()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_TRYLOCK_MUTEX,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A rwlock read-lock operation has been started.
 */
int _starpu_trace_rdlocking_rwlock()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_RDLOCKING_RWLOCK,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A rwlock read-lock operation has been completed.
 */
int _starpu_trace_rwlock_rdlocked()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_RWLOCK_RDLOCKED,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A rwlock write-lock operation has been started.
 */
int _starpu_trace_wrlocking_rwlock()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_WRLOCKING_RWLOCK,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A rwlock write-lock operation has been completed.
 */
int _starpu_trace_rwlock_wrlocked()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_RWLOCK_WRLOCKED,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A rwlock unlock operation has been started.
 */
int _starpu_trace_unlocking_rwlock()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_UNLOCKING_RWLOCK,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A rwlock unlock operation has been completed.
 */
int _starpu_trace_rwlock_unlocked()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_RWLOCK_UNLOCKED,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}


#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
#define _STARPU_TRACE_SPINLOCK_CONDITION (starpu_worker_get_type(starpu_worker_get_id()) == STARPU_CUDA_WORKER)
#endif
#endif

/**
 * A spin-lock lock operation has been started.
 */
int _starpu_trace_spinlock_locked(const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	if (_STARPU_TRACE_SPINLOCK_CONDITION)
	{
		const char *xfile;
		xfile = strrchr(file,'/') + 1;
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_SPINLOCK_LOCKED,line,_starpu_gettid(),xfile);
	}
#endif
#endif
	return 0;
}

/**
 * A spin-lock lock operation has been started.
 */
int _starpu_trace_locking_spinlock(const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	if (_STARPU_TRACE_SPINLOCK_CONDITION)
	{
		const char *xfile;
		xfile = strrchr(file,'/') + 1;
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_LOCKING_SPINLOCK,line,_starpu_gettid(),xfile);
	}
#endif
#endif
	return 0;
}

/**
 * A spin-lock unlock operation has been started.
 */
int _starpu_trace_unlocking_spinlock(const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	if (_STARPU_TRACE_SPINLOCK_CONDITION)
	{
		const char *xfile;
		xfile = strrchr(file,'/') + 1;
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_UNLOCKING_SPINLOCK,line,_starpu_gettid(),xfile);
	}
#endif
#endif
	return 0;
}

/**
 * A spin-lock unlock operation has been completed.
 */
int _starpu_trace_spinlock_unlocked(const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	if (_STARPU_TRACE_SPINLOCK_CONDITION)
	{
		const char *xfile;
		xfile = strrchr(file,'/') + 1;
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_SPINLOCK_UNLOCKED,line,_starpu_gettid(),xfile);
	}
#endif
#endif
	return 0;
}

/**
 * A spin-lock trylock operation has been attempted.
 */
int _starpu_trace_trylock_spinlock(const char *file STARPU_ATTRIBUTE_UNUSED, int line STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	if (_STARPU_TRACE_SPINLOCK_CONDITION)
	{
		const char *xfile;
		xfile = strrchr(file,'/') + 1;
		_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK_VERBOSE, _STARPU_FUT_TRYLOCK_SPINLOCK,line,_starpu_gettid(),xfile);
	}
#endif
#endif
	return 0;
}

/**
 * A wait operation on a condition has been started.
 */
int _starpu_trace_cond_wait_begin()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_COND_WAIT_BEGIN,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A wait operation on a condition has been completed.
 */
int _starpu_trace_cond_wait_end()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_COND_WAIT_END,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A thread has entered a barrier.
 */
int _starpu_trace_barrier_wait_begin()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_BARRIER_WAIT_BEGIN,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

/**
 * A thread has left a barrier.
 */
int _starpu_trace_barrier_wait_end()
{
#ifdef STARPU_FXT_LOCK_TRACES
#ifdef STARPU_USE_FXT
	const char *file;
	file = strrchr(__FILE__,'/') + 1;
	_STARPU_FUT_FULL_PROBE2STR(_STARPU_FUT_KEYMASK_LOCK, _STARPU_FUT_BARRIER_WAIT_END,__LINE__,_starpu_gettid(),file);
#endif
#endif
	return 0;
}

int _starpu_trace_data_load(int workerid STARPU_ATTRIBUTE_UNUSED, size_t size STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_DATA_LOAD, workerid, size, _starpu_gettid());
#endif
	return 0;
}

int _starpu_trace_start_unpartition(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned memnode STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_START_UNPARTITION_ON_TID, memnode, _starpu_gettid(), handle);
#endif
	return 0;
}

int _starpu_trace_end_unpartition(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned memnode STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE3(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_END_UNPARTITION_ON_TID, memnode, _starpu_gettid(), handle);
#endif
	return 0;
}

int _starpu_trace_sched_component_push_prio(struct starpu_sched_component *component STARPU_ATTRIBUTE_UNUSED, unsigned ntasks STARPU_ATTRIBUTE_UNUSED, double exp_len STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if (fut_active)
	{
		int workerid = STARPU_NMAXWORKERS + 1;
		if((component->nchildren == 1) && starpu_sched_component_is_worker(component->children[0]))
			workerid = starpu_sched_component_worker_get_workerid(component->children[0]);
		FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_PUSH_PRIO, _starpu_gettid(), workerid, ntasks, exp_len);
	}
#endif
	return 0;
}

int _starpu_trace_sched_component_pop_prio(struct starpu_sched_component *component STARPU_ATTRIBUTE_UNUSED, unsigned ntasks STARPU_ATTRIBUTE_UNUSED, double exp_len STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if (fut_active)
	{
		int workerid = STARPU_NMAXWORKERS + 1;
		if((component->nchildren == 1) && starpu_sched_component_is_worker(component->children[0]))
			workerid = starpu_sched_component_worker_get_workerid(component->children[0]);
		FUT_FULL_PROBE4(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_POP_PRIO, _starpu_gettid(), workerid, ntasks, exp_len);
	}
#endif
	return 0;
}

int _starpu_trace_sched_component_new(struct starpu_sched_component *component STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if (STARPU_UNLIKELY(fut_active)) _STARPU_FUT_ALWAYS_PROBE1STR(_STARPU_FUT_SCHED_COMPONENT_NEW, component, (component)->name);
#endif
	return 0;
}

int _starpu_trace_sched_component_connect(struct starpu_sched_component *parent STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_component *child STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if (STARPU_UNLIKELY(fut_active)) FUT_RAW_ALWAYS_PROBE2(FUT_CODE(_STARPU_FUT_SCHED_COMPONENT_CONNECT,2), parent, child);
#endif
	return 0;
}

int _starpu_trace_sched_component_push(struct starpu_sched_component *from STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_component *to STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED, int prio STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_PUSH, _starpu_gettid(), from, to, task, prio);
#endif
	return 0;
}

int _starpu_trace_sched_component_pull(struct starpu_sched_component *from STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_component *to STARPU_ATTRIBUTE_UNUSED, struct starpu_task *task STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE5(_STARPU_FUT_KEYMASK_SCHED, _STARPU_FUT_SCHED_COMPONENT_PULL, _starpu_gettid(), from, to, task, (task)->priority);
#endif
	return 0;
}

int _starpu_trace_handle_data_register(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	if(STARPU_UNLIKELY((_STARPU_FUT_KEYMASK_META) & fut_active))
	{
		const size_t __data_size = handle->ops->get_size(handle);
		const starpu_ssize_t __max_data_size = _starpu_data_get_max_size(handle);
		char __buf[(FXT_MAX_PARAMS-4)*sizeof(long)];
		void *__interface = handle->per_node[0].data_interface;
		if (handle->ops->describe)
			handle->ops->describe(__interface, __buf, sizeof(__buf));
		else
			__buf[0] = 0;
		_STARPU_FUT_FULL_PROBE5STR(_STARPU_FUT_KEYMASK_META, _STARPU_FUT_HANDLE_DATA_REGISTER, handle, __data_size, __max_data_size, handle->home_node, handle->parent_handle, __buf);
	}
#endif
	return 0;
}

int _starpu_trace_handle_data_unregister(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE1(_STARPU_FUT_KEYMASK_DATA, _STARPU_FUT_HANDLE_DATA_UNREGISTER, handle);
#endif
	return 0;
}

//Coherency Data Traces
int _starpu_trace_data_state_invalid(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_STATE_INVALID, handle, node);
#endif
	return 0;
}

int _starpu_trace_data_state_owner(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_STATE_OWNER, handle, node);
#endif
	return 0;
}

int _starpu_trace_data_state_shared(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, unsigned node STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_STATE_SHARED, handle, node);
#endif
	return 0;
}

int _starpu_trace_data_request_created(starpu_data_handle_t handle STARPU_ATTRIBUTE_UNUSED, int orig STARPU_ATTRIBUTE_UNUSED, int dest STARPU_ATTRIBUTE_UNUSED, int prio STARPU_ATTRIBUTE_UNUSED, enum starpu_is_prefetch is_prefetch STARPU_ATTRIBUTE_UNUSED, struct _starpu_data_request *req STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE6(_STARPU_FUT_KEYMASK_DSM_VERBOSE, _STARPU_FUT_DATA_REQUEST_CREATED, orig, dest, prio, handle, is_prefetch, req);
#endif
	return 0;
}

int _starpu_trace_memory_full(size_t size STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	FUT_FULL_PROBE2(_STARPU_FUT_KEYMASK_DSM, _STARPU_FUT_MEMORY_FULL,size,_starpu_gettid());
#endif
	return 0;
}

int _starpu_trace_start_transfer(unsigned memnode STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_starpu_trace_start_progress(memnode, worker);
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer)
	{
		struct starpu_prof_tool_info pi;
		enum starpu_prof_tool_driver_type driver_type;
		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			driver_type = starpu_prof_tool_driver_cpu;
			break;
		case STARPU_CUDA_WORKER:
		case STARPU_HIP_WORKER:
		case STARPU_OPENCL_WORKER:
			driver_type = starpu_prof_tool_driver_gpu;
			break;
		default:
			goto out;
		}

		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_start_transfer, worker->workerid, worker->workerid, driver_type, memnode, worker->nb_buffers_totransfer, worker->nb_buffers_transferred);

		starpu_prof_tool_callbacks.starpu_prof_tool_event_start_transfer(&pi, NULL, NULL);
	out:
		;
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS
//	uint64_t myguid = 0;// //new_guid(); TODO

	tasktimer_execution_space_t source_info, dest_info; /* TODO will set that later */
	tasktimer_execution_space_p sip = &source_info;
	tasktimer_execution_space_p dip = &dest_info;
	source_info.type = TASKTIMER_DEVICE_CPU;
	source_info.device_id = 0;
	source_info.instance_id = 0;
	dest_info.type = TASKTIMER_DEVICE_CPU;
	dest_info.device_id = 0;
	dest_info.instance_id = 0;

	char *source = &memnode;
	char *dest = &memnode;/* TODO will set that later */

	// TASKTIMER_DATA_TRANSFER_START(myguid, sip, "source", (void*)source, dip, "dest", (void*)dest);
#endif
return 0;
}

int _starpu_trace_end_transfer(unsigned memnode STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_starpu_trace_end_progress(memnode, worker);
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer)
	{
		struct starpu_prof_tool_info pi;
		enum starpu_prof_tool_driver_type driver_type;
		switch(worker->arch)
		{
		case STARPU_CPU_WORKER:
			driver_type = starpu_prof_tool_driver_cpu;
			break;
		case STARPU_CUDA_WORKER:
		case STARPU_HIP_WORKER:
		case STARPU_OPENCL_WORKER:
			driver_type = starpu_prof_tool_driver_gpu;
			break;
		default:
			goto out;
		}

//		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_transfer, workerid, workerid, starpu_prof_tool_driver_cpu, memnode, NULL);
		pi = _starpu_prof_tool_get_info_d(starpu_prof_tool_event_end_transfer, worker->workerid, worker->workerid, driver_type, memnode, worker->nb_buffers_totransfer, worker->nb_buffers_transferred);

		/* pi.model_name = _starpu_job_get_model_name(j);
		   pi.task_name = _starpu_job_get_task_name(j); */
		starpu_prof_tool_callbacks.starpu_prof_tool_event_end_transfer(&pi, NULL, NULL);
	out:
		;
	}
#endif

#ifdef STARPU_PROF_TASKSTUBS
//	TASKTIMER_DATA_TRANSFER_STOP(100); /* TODO */
#endif
	return 0;
}

/**
 * A worker thread initialization has been started.
 * \p archtype is the architecture type.
 * \p sync is unused.
 */
int _starpu_trace_worker_init_start(struct _starpu_worker *worker STARPU_ATTRIBUTE_UNUSED,
				   enum starpu_worker_archtype archtype STARPU_ATTRIBUTE_UNUSED,
				   unsigned sync STARPU_ATTRIBUTE_UNUSED)
{
	unsigned devid = worker->devid;
	unsigned memnode = worker->memory_node;
	(void) devid;
	(void) memnode;

#ifdef STARPU_USE_FXT
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBE7(_STARPU_FUT_WORKER_INIT_START, _STARPU_FUT_WORKER_KEY(archtype), worker->workerid, devid, memnode, worker->bindid, sync, _starpu_gettid());
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start)
	{
		enum starpu_prof_tool_driver_type drivertype;
		switch(archtype)
		{
		case STARPU_CPU_WORKER: drivertype = starpu_prof_tool_driver_cpu; break;
		case STARPU_CUDA_WORKER: drivertype = starpu_prof_tool_driver_gpu; break;
		case STARPU_OPENCL_WORKER: drivertype = starpu_prof_tool_driver_ocl; break;
		case STARPU_HIP_WORKER: drivertype = starpu_prof_tool_driver_hip; break;
		default: drivertype = starpu_prof_tool_driver_cpu; break;
		}

		struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init, devid, worker->workerid, drivertype, -1, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init(&pi, NULL, NULL);

		pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_driver_init_start, devid, worker->workerid, drivertype, -1, NULL);
		starpu_prof_tool_callbacks.starpu_prof_tool_event_driver_init_start(&pi, NULL, NULL);
	}
#endif

	return 0;
}

/**
 * A worker thread initialization has been completed.
 * \p workerid is the id of the worker.
 */
int _starpu_trace_worker_init_end(struct _starpu_worker *worker  STARPU_ATTRIBUTE_UNUSED,
				 enum starpu_worker_archtype archtype STARPU_ATTRIBUTE_UNUSED)
{
	/* todo: replace starpu_prof_tool_driver_type with enum starpu_worker_archtype to make the API consistent ? */
#ifdef STARPU_USE_FXT
	if (_starpu_fxt_started)
		FUT_DO_ALWAYS_PROBE2(_STARPU_FUT_WORKER_INIT_END, _starpu_gettid(), worker->workerid);
#endif

#ifdef STARPU_PROF_TOOL
	if(starpu_prof_tool_callbacks.starpu_prof_tool_event_init_end)
	{
		enum starpu_prof_tool_driver_type driver_type;
		switch(archtype)
		{
		case STARPU_CPU_WORKER: driver_type = starpu_prof_tool_driver_cpu; break;
		case STARPU_CUDA_WORKER: driver_type = starpu_prof_tool_driver_gpu; break;
		case STARPU_OPENCL_WORKER: driver_type = starpu_prof_tool_driver_ocl; break;
		case STARPU_HIP_WORKER: driver_type = starpu_prof_tool_driver_hip; break;
		default: driver_type = starpu_prof_tool_driver_cpu; break;
		}

		struct starpu_prof_tool_info pi = _starpu_prof_tool_get_info_init(starpu_prof_tool_event_init_end, 0, driver_type, &(_starpu_config.conf));
		pi.conf = &_starpu_config.conf;
		starpu_prof_tool_callbacks.starpu_prof_tool_event_init_end(&pi, NULL, NULL);
	}
#endif

	return 0;
}
