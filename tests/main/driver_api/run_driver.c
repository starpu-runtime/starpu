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
#include <unistd.h>

#include "../../helper.h"

/*
 * Users can directly control drivers by using the starpu_driver* functions.
 *
 * This test makes sure that the starpu_driver_run function works for CPU, CUDA
 * and OpenCL drivers, and that the starpu_drivers_request_termination function
 * correctly shuts down all drivers.
 *
 * The test_* functions can return:
 * - 0 (success)
 * - 1 (failure)
 * - STARPU_TEST_SKIPPED (non-critical errors)
 */

#if defined(STARPU_USE_CPU) || defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
static void dummy(void *buffers[], void *args)
{
	(void) buffers;
	(*(int *)args)++;
	usleep(100000);
}

static struct starpu_codelet cl =
{
	.cpu_funcs    = { dummy },
	.cuda_funcs   = { dummy },
	.opencl_funcs = { dummy },
	.nbuffers     = 0
};

static void *run_driver(void *arg)
{
	struct starpu_driver *d = (struct starpu_driver *) arg;
	int ret = starpu_driver_run(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_run");
	return NULL;
}
#endif /* STARPU_USE_CPU || STARPU_USE_CUDA || STARPU_USE_OPENCL */

#ifdef STARPU_USE_CPU
static int test_cpu(void)
{
	int ret, var = 0;
	static starpu_pthread_t driver_thread;
	struct starpu_conf conf;
	struct starpu_driver d =
	{
		.type = STARPU_CPU_WORKER,
		.id.cpu_id = 0
	};

	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.n_not_launched_drivers = 1;
	conf.not_launched_drivers = &d;
	conf.ncpus = 1;
	conf.ncuda = 0;
	conf.nopencl = 0;
	ret = starpu_init(&conf);
	if (ret == -ENODEV || starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "WARNING: No CPU worker found\n");
		if (ret == 0)
			starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_pthread_create(&driver_thread, NULL, run_driver, &d);
	if (ret != 0)
	{
		ret = 1;
		goto out2;
	}

	struct starpu_task *task;
	task = starpu_task_create();
	cl.where = STARPU_CPU;
	task->cl = &cl;
	task->cl_arg = &var;
	task->synchronous = 1;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "WARNING: No worker can execute this task\n");
		ret = STARPU_TEST_SKIPPED;
		goto out;
	}

	FPRINTF(stderr, "[CPU] Var = %d (expected value: 1)\n", var);
	ret = !!(var != 1);
out:
	starpu_drivers_request_termination();
	if (starpu_pthread_join(driver_thread, NULL) != 0)
		return 1;
out2:
	starpu_shutdown();
	return ret;
}
#endif /* STARPU_USE_CPU */

#ifdef STARPU_USE_CUDA
static int test_cuda(void)
{
	int ret, var = 0;
	static starpu_pthread_t driver_thread;
	struct starpu_conf conf;
	struct starpu_driver d =
	{
		.type = STARPU_CUDA_WORKER,
		.id.cuda_id = 0
	};

	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.n_not_launched_drivers = 1;
	conf.not_launched_drivers = &d;
	conf.ncpus = 0;
	conf.ncuda = 1;
	conf.nopencl = 0;
	ret = starpu_init(&conf);
	if (ret == -ENODEV || starpu_cuda_worker_get_count() == 0)
	{
		FPRINTF(stderr, "WARNING: No CUDA worker found\n");
		if (ret == 0)
			starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}
	if (starpu_cuda_worker_get_count() > 1)
	{
		FPRINTF(stderr, "WARNING: More than one worker, this is not supported by this test\n");
		if (ret == 0)
			starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_pthread_create(&driver_thread, NULL, run_driver, &d);
	if (ret == -1)
		goto out;

	struct starpu_task *task;
	task = starpu_task_create();
	cl.where = STARPU_CUDA;
	task->cl = &cl;
	task->cl_arg = &var;
	task->synchronous = 1;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "WARNING: No worker can execute this task\n");
		goto out;
	}

out:
	starpu_drivers_request_termination();
	if (starpu_pthread_join(driver_thread, NULL) != 0)
		return 1;
	starpu_shutdown();

	FPRINTF(stderr, "[CUDA] Var = %d (expected value: 1)\n", var);
	ret = !!(var != 1);
	return ret;
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_OPENCL
static int test_opencl(void)
{
	int ret, var = 0;
	static starpu_pthread_t driver_thread;
	struct starpu_conf conf;

	cl_int err;
        cl_uint pdummy;
        cl_platform_id platform;
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

	struct starpu_driver d =
	{
		.type = STARPU_OPENCL_WORKER,
		.id.opencl_id = device_id
	};

	starpu_conf_init(&conf);
	conf.precedence_over_environment_variables = 1;
	conf.n_not_launched_drivers = 1;
	conf.not_launched_drivers = &d;
	conf.ncpus = 1;
	conf.ncuda = 0;
	conf.nopencl = 1;
	ret = starpu_init(&conf);
	if (ret == -ENODEV || starpu_opencl_worker_get_count() == 0)
	{
		FPRINTF(stderr, "WARNING: No OpenCL workers found\n");
		if (ret == 0)
			starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_pthread_create(&driver_thread, NULL, run_driver, &d);
	if (ret == -1)
		goto out;

	struct starpu_task *task;
	task = starpu_task_create();
	cl.where = STARPU_OPENCL;
	task->cl = &cl;
	task->cl_arg = &var;
	task->synchronous = 1;

	ret = starpu_task_submit(task);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "WARNING: No worker can execute the task\n");
		goto out;
	}

out:
	starpu_drivers_request_termination();
	if (starpu_pthread_join(driver_thread, NULL) != 0)
		return 1;
	starpu_shutdown();

	FPRINTF(stderr, "[OpenCL] Var = %d (expected value: 1)\n", var);
	ret = !!(var != 1);
	return ret;
}
#endif /* STARPU_USE_OPENCL */

int main(void)
{
	int ret = STARPU_TEST_SKIPPED;

#ifdef STARPU_USE_CPU
	ret = test_cpu();
	if (ret == 1)
		return 1;
#endif
#ifdef STARPU_USE_CUDA
	ret = test_cuda();
	if (ret == 1)
		return 1;
#endif
#ifdef STARPU_USE_OPENCL
	ret = test_opencl();
	if (ret == 1)
		return 1;
#endif
	return ret;
}
