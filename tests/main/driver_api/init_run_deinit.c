/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include "../../helper.h"

#define NTASKS 8

#if defined(STARPU_USE_CPU) || defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
void dummy(void *buffers[], void *args)
{
	(void) buffers;
	(*(int *)args)++;
}

static struct starpu_codelet cl =
{
	.cpu_funcs    = { dummy },
	.cuda_funcs   = { dummy },
	.opencl_funcs = { dummy },
	.nbuffers     = 0
};

static void
init_driver(struct starpu_driver *d)
{
	int ret;
	ret = starpu_driver_init(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_init");
}

static void
run(struct starpu_task *task, struct starpu_driver *d)
{
	int ret;
	ret = starpu_task_submit(task);
	starpu_do_schedule();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	while (!starpu_task_finished(task))
	{
		ret = starpu_driver_run_once(d);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_run_once");
	}
	ret = starpu_task_wait(task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait");
}

static void
deinit_driver(struct starpu_driver *d)
{
	int ret;
	ret = starpu_driver_deinit(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_deinit");
}
#endif /* STARPU_USE_CPU || STARPU_USE_CUDA || STARPU_USE_OPENCL */

#ifdef STARPU_USE_CPU
static int test_cpu(void)
{
	int var = 0, ret, ncpu;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;

	struct starpu_driver d =
	{
		.type = STARPU_CPU_WORKER,
		.id.cpu_id = 0
	};

	conf.precedence_over_environment_variables = 1;
	conf.ncpus = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "WARNING: No CPU worker found\n");
		return STARPU_TEST_SKIPPED;
	}

	ncpu = starpu_cpu_worker_get_count();
	if (ncpu == 0)
	{
		FPRINTF(stderr, "WARNING: No CPU worker found\n");
		return STARPU_TEST_SKIPPED;
	}

	init_driver(&d);
	int i;
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task;
		task = starpu_task_create();
		cl.where = STARPU_CPU;
		task->cl = &cl;
		task->cl_arg = &var;
		task->detach = 0;

		run(task, &d);
	}
	deinit_driver(&d);

	starpu_task_wait_for_all();
	starpu_shutdown();

	FPRINTF(stderr, "[CPU] Var is %d (expected value: %d)\n", var, NTASKS);
	return !!(var != NTASKS);
}
#endif /* STARPU_USE_CPU */

#ifdef STARPU_USE_CUDA
static int test_cuda(void)
{
	int var = 0, ret, ncuda;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;

	struct starpu_driver d =
	{
		.type = STARPU_CUDA_WORKER,
		.id.cuda_id = 0
	};

	conf.precedence_over_environment_variables = 1;
	conf.ncpus = 0;
	conf.ncuda = 1;
	conf.nopencl = 0;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "WARNING: No CUDA worker found\n");
		return STARPU_TEST_SKIPPED;
	}

	ncuda = starpu_cuda_worker_get_count();
	if (ncuda == 0)
	{
		FPRINTF(stderr, "WARNING: No CUDA worker found\n");
		return STARPU_TEST_SKIPPED;
	}

	init_driver(&d);
	int i;
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task;
		task = starpu_task_create();
		cl.where = STARPU_CUDA;
		task->cl = &cl;
		task->cl_arg = &var;
		task->detach = 0;

		run(task, &d);
	}
	deinit_driver(&d);

	starpu_task_wait_for_all();
	starpu_shutdown();

	FPRINTF(stderr, "[CUDA] Var is %d (expected value: %d)\n", var, NTASKS);
	return !!(var != NTASKS);
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int test_opencl(void)
{
        cl_int err;
        cl_platform_id platform;
        cl_uint pdummy;
	int nopencl;

        err = clGetPlatformIDs(1, &platform, &pdummy);
        if (err != CL_SUCCESS)
        {
		FPRINTF(stderr, "WARNING: No OpenCL platform found\n");
		return STARPU_TEST_SKIPPED;
	}

	cl_device_type device_type = CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_ACCELERATOR;
	if (starpu_get_env_number("STARPU_OPENCL_ON_CPUS") > 0)
		device_type |= CL_DEVICE_TYPE_CPU;
	if (starpu_get_env_number("STARPU_OPENCL_ONLY_ON_CPUS") > 0)
		device_type = CL_DEVICE_TYPE_CPU;

	cl_device_id device_id;
        err = clGetDeviceIDs(platform, device_type, 1, &device_id, NULL);
        if (err != CL_SUCCESS)
        {
		FPRINTF(stderr, "WARNING: No GPU devices found on OpenCL platform\n");
		return STARPU_TEST_SKIPPED;
	}

	int var = 0, ret;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;

	struct starpu_driver d =
	{
		.type = STARPU_OPENCL_WORKER,
		.id.opencl_id = device_id
	};

	conf.precedence_over_environment_variables = 1;
	conf.ncpus = 0;
	conf.ncuda = 0;
	conf.nopencl = 1;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "WARNING: No OpenCL workers found\n");
		return STARPU_TEST_SKIPPED;
	}

	nopencl = starpu_opencl_worker_get_count();
	if (nopencl == 0)
	{
		FPRINTF(stderr, "WARNING: No OpenCL workers found\n");
		return STARPU_TEST_SKIPPED;
	}

	init_driver(&d);
	int i;
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task;
		task = starpu_task_create();
		cl.where = STARPU_OPENCL;
		task->cl = &cl;
		task->cl_arg = &var;
		task->detach = 0;

		run(task, &d);
	}
	deinit_driver(&d);


	starpu_task_wait_for_all();
	starpu_shutdown();

	FPRINTF(stderr, "[OpenCL] Var is %d (expected value: %d)\n", var, NTASKS);
	return !!(var != NTASKS);
}
#endif /* STARPU_USE_OPENCL */

int main(void)
{
	int ret = STARPU_TEST_SKIPPED;

#ifdef STARPU_USE_CPU
	ret = test_cpu();
	if (ret == 1)
		return ret;
#endif
#ifdef STARPU_USE_CUDA
	ret = test_cuda();
	if (ret == 1)
		return ret;
#endif
#ifdef STARPU_USE_OPENCL
	ret = test_opencl();
	if (ret == 1)
		return ret;
#endif

	return ret;
}

