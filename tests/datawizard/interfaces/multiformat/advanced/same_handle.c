/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "generic.h"
#include "../../../../helper.h"

/*
 * A single handle can be given twice to a given kernel. In this case, it
 * should only be converted once.
 */
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
extern struct stats global_stats;
static int vector[NX]; static starpu_data_handle_t handle;

static struct starpu_codelet cl =
{
	.modes = { STARPU_RW, STARPU_RW },
#ifdef STARPU_USE_CUDA
	.cuda_funcs = { cuda_func },
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = { opencl_func },
#endif
	.nbuffers = 2,
};

static void
register_handle(void)
{
	int i;
	for (i = 0; i < NX; i++)
		vector[i] = i;
	starpu_multiformat_data_register(&handle, STARPU_MAIN_RAM, vector, NX, &ops);
}

static void
unregister_handle(void)
{
	starpu_data_unregister(handle);
}

static int
create_and_submit_tasks(void)
{
	int ret;
	struct starpu_task *task;

	cl.where = 0;
#ifdef STARPU_USE_CUDA
	cl.where |= STARPU_CUDA;
#endif
#ifdef STARPU_USE_OPENCL
	cl.where |= STARPU_OPENCL;
#endif

	task = starpu_task_create();
	task->cl = &cl;
	task->handles[0] = handle;
	task->handles[1] = handle;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV)
	{
		task->destroy = 0;
		starpu_task_destroy(task);
		return -ENODEV;
	}

	return 0;
}
#endif

int
main(int argc, char **argv)
{
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	int err;

	struct starpu_conf conf;
	starpu_conf_init(&conf);
#if defined(STARPU_USE_CUDA0) || defined(STARPU_USE_CUDA1)
	conf.ncuda = 0;
#endif
	err = starpu_initialize(&conf, &argc, &argv);
	if (err == -ENODEV)
		goto enodev;

	reset_stats(&global_stats);
	register_handle();
	err = create_and_submit_tasks();
	unregister_handle();
	starpu_shutdown();

	if (err == -ENODEV)
		goto enodev;

#ifdef STARPU_USE_CUDA
	if (global_stats.cuda == 1)
	{
		if (global_stats.cpu_to_cuda == 1 &&
		    global_stats.cuda_to_cpu == 1)
			return EXIT_SUCCESS;
		else
			return EXIT_FAILURE;
	}
	else
#endif
#if defined(STARPU_USE_OPENCL)
	if (global_stats.opencl == 1)
	{
		if (global_stats.cpu_to_opencl == 1 &&
		    global_stats.opencl_to_cpu == 1)
			return EXIT_SUCCESS;
		else
			return EXIT_FAILURE;

	}
	else
#endif
	{
		/* We should not get here */
		return EXIT_FAILURE;
	}

enodev:
#endif
	return STARPU_TEST_SKIPPED;
}
