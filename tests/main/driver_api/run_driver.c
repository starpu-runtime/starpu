/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#if defined(STARPU_USE_CPU) || defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_USE_HIP)
static void dummy(void *buffers[], void *args)
{
	(void) buffers;
	(*(int *)args)++;
	starpu_usleep(100000);
}

static struct starpu_codelet cl =
{
	.cpu_funcs    = { dummy },
	.cuda_funcs   = { dummy },
	.opencl_funcs = { dummy },
	.hip_funcs    = { dummy },
	.nbuffers     = 0
};

static void *run_driver(void *arg)
{
	struct starpu_driver *d = (struct starpu_driver *) arg;
	int ret = starpu_driver_run(d);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_driver_run");
	return NULL;
}

typedef unsigned (*worker_get_count)(void);

static int test_driver(struct starpu_conf *conf, struct starpu_driver *d, const char *name_driver, worker_get_count worker_get_count_func, int32_t where_driver)
{
	int ret, var = 0;
	static starpu_pthread_t driver_thread;

	ret = starpu_init(conf);
	if (ret == -ENODEV || worker_get_count_func() == 0)
	{
		FPRINTF(stderr, "WARNING: No %s worker found\n", name_driver);
		if (ret == 0)
			starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	ret = starpu_pthread_create(&driver_thread, NULL, run_driver, d);
	if (ret != 0)
	{
		ret = 1;
		goto out2;
	}

	struct starpu_task *task;
	task = starpu_task_create();
	cl.where = where_driver;
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

	FPRINTF(stderr, "[%s] Var = %d (expected value: 1)\n", name_driver, var);
	ret = !!(var != 1);
out:
	starpu_drivers_request_termination();
	if (starpu_pthread_join(driver_thread, NULL) != 0)
		return 1;
out2:
	starpu_shutdown();
	return ret;
}
#endif /* STARPU_USE_CPU || STARPU_USE_CUDA || STARPU_USE_OPENCL || STARPU_USE_HIP */

#ifdef STARPU_USE_CPU
static int test_cpu(void)
{
	int ret;
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
	starpu_conf_noworker(&conf);
	conf.ncpus = 1;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	return test_driver(&conf, &d, "CPU", starpu_cpu_worker_get_count, STARPU_CPU);
}
#endif /* STARPU_USE_CPU */

#ifdef STARPU_USE_CUDA
static int test_cuda(void)
{
	int ret;
	struct starpu_conf conf;
	int cudaid = 0;
	char *cudaid_str = getenv("STARPU_WORKERS_CUDAID");

	if (cudaid_str)
		cudaid = atoi(cudaid_str);

	/* FIXME: starpu_driver would need another field to specify which stream we're driving */
	if (starpu_getenv_number_default("STARPU_NWORKER_PER_CUDA", 1) != 1 &&
	    starpu_getenv_number_default("STARPU_CUDA_THREAD_PER_WORKER", -1) > 0)
		return STARPU_TEST_SKIPPED;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;

	struct starpu_driver d =
	{
		.type = STARPU_CUDA_WORKER,
		.id.cuda_id = cudaid
	};

	conf.precedence_over_environment_variables = 1;
	starpu_conf_noworker(&conf);
	conf.ncuda = 1;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	return test_driver(&conf, &d, "CUDA", starpu_cuda_worker_get_count, STARPU_CUDA);
}
#endif /* STARPU_USE_CUDA */

#ifdef STARPU_USE_HIP
static int test_hip(void)
{
	int ret;
	struct starpu_conf conf;

	ret = starpu_conf_init(&conf);
	if (ret == -EINVAL)
		return 1;

	struct starpu_driver d =
	{
		.type = STARPU_HIP_WORKER,
		.id.hip_id = 0
	};

	conf.precedence_over_environment_variables = 1;
	starpu_conf_noworker(&conf);
	conf.nhip = 1;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	return test_driver(&conf, &d, "HIP", starpu_hip_worker_get_count, STARPU_HIP);
}
#endif /* STARPU_USE_HIP */

#ifdef STARPU_USE_OPENCL
static int test_opencl(void)
{
	int ret;

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
	if (starpu_getenv_number("STARPU_OPENCL_ON_CPUS") > 0)
		device_type |= CL_DEVICE_TYPE_CPU;
	if (starpu_getenv_number("STARPU_OPENCL_ONLY_ON_CPUS") > 0)
		device_type = CL_DEVICE_TYPE_CPU;

	cl_device_id device_id;
        err = clGetDeviceIDs(platform, device_type, 1, &device_id, NULL);
        if (err != CL_SUCCESS)
	{
		FPRINTF(stderr, "WARNING: No GPU devices found on OpenCL platform\n");
		return STARPU_TEST_SKIPPED;
	}

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
	starpu_conf_noworker(&conf);
	conf.nopencl = 1;
	conf.not_launched_drivers = &d;
	conf.n_not_launched_drivers = 1;

	return test_driver(&conf, &d, "OpenCL", starpu_opencl_worker_get_count, STARPU_OPENCL);
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
#if defined(STARPU_USE_CUDA) && !(defined(STARPU_USE_CUDA0) || defined(STARPU_USE_CUDA1))
	ret = test_cuda();
	if (ret == 1)
		return 1;
#endif
#ifdef STARPU_USE_OPENCL
	ret = test_opencl();
	if (ret == 1)
		return 1;
#endif
#ifdef STARPU_USE_HIP
	ret = test_hip();
	if (ret == 1)
		return 1;
#endif
	return ret;
}
