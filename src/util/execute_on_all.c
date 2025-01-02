/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/config.h>
#include <core/jobs.h>
#include <core/task.h>
#include <core/workers.h>
#include <profiling/callbacks.h>

struct wrapper_func_args
{
	void (*func)(void *);
	void *arg;
};

static void wrapper_func(void *buffers[] STARPU_ATTRIBUTE_UNUSED, void *_args)
{
	struct wrapper_func_args *args = (struct wrapper_func_args *) _args;
#ifdef STARPU_PROF_TOOL
	struct starpu_prof_tool_info pi;
#endif

#ifdef STARPU_PROF_TOOL
	int worker = starpu_worker_get_id();
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_start_gpu_exec, worker, worker, starpu_prof_tool_driver_gpu, -1, (void*)args->func);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_start_gpu_exec(&pi, NULL, NULL);
#endif

	args->func(args->arg);

#ifdef STARPU_PROF_TOOL
	pi = _starpu_prof_tool_get_info(starpu_prof_tool_event_end_gpu_exec, worker, worker, starpu_prof_tool_driver_gpu, -1, (void*)args->func);
	starpu_prof_tool_callbacks.starpu_prof_tool_event_end_gpu_exec(&pi, NULL, NULL);
#endif
}

/**
 * Execute func(arg) on the given workers.
 */
void starpu_execute_on_specific_workers(void (*func)(void*), void * arg, unsigned num_workers, unsigned * workers, const char * name)
{
	int ret;
	unsigned w;
	struct starpu_task *tasks[STARPU_NMAXWORKERS];

	/* create a wrapper codelet */
	struct starpu_codelet wrapper_cl =
	{
		.where = 0xFF,
		.cuda_funcs = {wrapper_func},
		.hip_funcs = {wrapper_func},
		.cpu_funcs = {wrapper_func},
		.opencl_funcs = {wrapper_func},
		.nbuffers = 0,
		.name = name
	};

	struct wrapper_func_args args =
	{
		.func = func,
		.arg = arg
	};


	for (w = 0; w < num_workers; w++)
	{
		unsigned worker = workers[w];
		tasks[w] = starpu_task_create();
		tasks[w]->name = name;

		tasks[w]->cl = &wrapper_cl;
		tasks[w]->cl_arg = &args;

		tasks[w]->execute_on_a_specific_worker = 1;
		tasks[w]->workerid = worker;

		tasks[w]->detach = 0;
		tasks[w]->destroy = 0;

		_starpu_exclude_task_from_dag(tasks[w]);

		ret = starpu_task_submit(tasks[w]);
		if (ret == -ENODEV)
		{
			/* if the worker is not able to execute this tasks, we
			 * don't insist as this means the worker is not
			 * designated by the "where" bitmap */
			starpu_task_destroy(tasks[w]);
			tasks[w] = NULL;
		}
	}

	for (w= 0; w < num_workers; w++)
	{
		if (tasks[w])
		{
			ret = starpu_task_wait(tasks[w]);
			STARPU_ASSERT(!ret);
			starpu_task_destroy(tasks[w]);
		}
	}
}

/* execute func(arg) on each worker that matches the "where" flag */
void starpu_execute_on_each_worker_ex(void (*func)(void *), void *arg, uint32_t where, const char * name)
{
	int ret;
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	struct starpu_task *tasks[STARPU_NMAXWORKERS];

	STARPU_ASSERT_MSG((where & ~STARPU_CPU & ~STARPU_CUDA & ~STARPU_OPENCL & ~STARPU_HIP) == 0, "This function is implemented only on CPU, CUDA, HIP, OpenCL");

	/* create a wrapper codelet */
	struct starpu_codelet wrapper_cl =
	{
		.where = where,
		.cuda_funcs = {wrapper_func},
		.hip_funcs = {wrapper_func},
		.cpu_funcs = {wrapper_func},
		.opencl_funcs = {wrapper_func},
		.nbuffers = 0,
		.name = (name != NULL ? name : "execute_on_all_wrapper")
	};

	struct wrapper_func_args args =
	{
		.func = func,
		.arg = arg
	};


	for (worker = 0; worker < nworkers; worker++)
	{
		tasks[worker] = starpu_task_create();
		tasks[worker]->name = wrapper_cl.name;

		tasks[worker]->cl = &wrapper_cl;
		tasks[worker]->cl_arg = &args;

		tasks[worker]->execute_on_a_specific_worker = 1;
		tasks[worker]->workerid = worker;

		tasks[worker]->detach = 0;
		tasks[worker]->destroy = 0;

		_starpu_exclude_task_from_dag(tasks[worker]);

		ret = _starpu_task_submit_internally(tasks[worker]);
		if (ret == -ENODEV)
		{
			/* if the worker is not able to execute this task, we
			 * don't insist as this means the worker is not
			 * designated by the "where" bitmap */
			starpu_task_destroy(tasks[worker]);
			tasks[worker] = NULL;
		}
	}

	for (worker = 0; worker < nworkers; worker++)
	{
		if (tasks[worker])
		{
			ret = starpu_task_wait(tasks[worker]);
			STARPU_ASSERT(!ret);
			starpu_task_destroy(tasks[worker]);
		}
	}
}

void starpu_execute_on_each_worker(void (*func)(void *), void *arg, uint32_t where)
{
	starpu_execute_on_each_worker_ex(func, arg, where, NULL);
}
