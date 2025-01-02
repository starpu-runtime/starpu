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

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <starpu.h>
#include <stdlib.h>
#include "../helper.h"

/*
 * Try calling starpu_data_prefetch_on_node before running a task there
 */

#ifdef STARPU_QUICK_CHECK
#define N 10
#elif !defined(STARPU_LONG_CHECK)
#define N 100
#else
#define N 1000
#endif

#define VECTORSIZE	1024

starpu_data_handle_t v_handle;
static unsigned *v;

static
void callback(void *arg)
{
	unsigned node = (unsigned)(uintptr_t) arg;

	starpu_data_prefetch_on_node(v_handle, node, 1);
}

void codelet_null(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
}

static struct starpu_codelet cl_r =
{
	.cpu_funcs = {codelet_null},
	.cuda_funcs = {codelet_null},
	.opencl_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

static struct starpu_codelet cl_w =
{
	.cpu_funcs = {codelet_null},
	.cuda_funcs = {codelet_null},
	.opencl_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

static struct starpu_codelet cl_rw =
{
	.cpu_funcs = {codelet_null},
	.cuda_funcs = {codelet_null},
	.opencl_funcs = {codelet_null},
	.cpu_funcs_name = {"codelet_null"},
	.nbuffers = 1,
	.modes = {STARPU_RW}
};

static struct starpu_codelet *select_codelet_with_random_mode(void)
{
	int r = rand();

	switch (r % 3)
	{
		case 0:
			return &cl_r;
		case 1:
			return &cl_w;
		case 2:
			return &cl_rw;
	};
	return &cl_rw;
}

int main(int argc, char **argv)
{
	int ret;

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	starpu_conf_noworker(&conf);
	
	conf.ncpus = -1;
	conf.ncuda = -1;
	conf.nopencl = -1;
	conf.nmpi_ms = -1;
	conf.ntcpip_ms = -1;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&v, VECTORSIZE*sizeof(unsigned));
	starpu_vector_data_register(&v_handle, STARPU_MAIN_RAM, (uintptr_t)v, VECTORSIZE, sizeof(unsigned));

	unsigned nworker = starpu_worker_get_count();

	unsigned iter, worker;
	for (iter = 0; iter < N; iter++)
	{
		for (worker = 0; worker < nworker; worker++)
		{
			/* synchronous prefetch */
			unsigned node = starpu_worker_get_memory_node(worker);
			ret = starpu_data_prefetch_on_node(v_handle, node, 0);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_prefetch_on_node");

			/* execute a task */
			struct starpu_task *task = starpu_task_create();

			task->handles[0] = v_handle;
			task->cl = select_codelet_with_random_mode();
			task->synchronous = 1;
			task->execute_on_a_specific_worker = 1;
			task->workerid = worker;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}

	for (iter = 0; iter < N; iter++)
	{
		for (worker = 0; worker < nworker; worker++)
		{
			/* asynchronous prefetch */
			unsigned node = starpu_worker_get_memory_node(worker);
			ret = starpu_data_prefetch_on_node(v_handle, node, 1);
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_data_prefetch_on_node");

			/* execute a task */
			struct starpu_task *task = starpu_task_create();

			task->handles[0] = v_handle;
			task->cl = select_codelet_with_random_mode();
			task->callback_func = callback;
			task->callback_arg = (void*)(uintptr_t) starpu_worker_get_memory_node((worker+1)%nworker);
			task->execute_on_a_specific_worker = 1;
			task->workerid = worker;

			task->synchronous = 0;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	starpu_data_unregister(v_handle);
	starpu_free_noflag(v, VECTORSIZE*sizeof(unsigned));
	starpu_shutdown();

	return EXIT_SUCCESS;

enodev:
	starpu_free_noflag(v, VECTORSIZE*sizeof(unsigned));
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
