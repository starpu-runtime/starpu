/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
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
#include <starpu_scheduler.h>
#include "../helper.h"

#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  return(retval); }

/*
 * Benchmark memset with a linear and non-linear regression
 */

#define STARTlin 1024
#define START 1024
#ifdef STARPU_QUICK_CHECK
#define END 1048576
#define NENERGY 3
#else
#define END 16777216
#define NENERGY 100
#endif

#ifdef STARPU_USE_CUDA
static void memset_cuda(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	cudaMemsetAsync(ptr, 42, n * sizeof(*ptr), starpu_cuda_get_local_stream());
}
#endif

#ifdef STARPU_USE_OPENCL
extern void memset_opencl(void *buffers[], void *args);
#endif

void memset0_cpu(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
	unsigned i;

	for (i = 0; i < n; i++)
		ptr[i] = 42;
}

void memset_cpu(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	starpu_usleep(10);
	memset(ptr, 42, n * sizeof(*ptr));
}

static struct starpu_perfmodel model =
{
	.type = STARPU_REGRESSION_BASED,
	.symbol = "memset_regression_based"
};

static struct starpu_perfmodel nl_model =
{
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "non_linear_memset_regression_based"
};

static struct starpu_perfmodel energy_model =
{
	.type = STARPU_REGRESSION_BASED,
	.symbol = "memset_regression_based_energy"
};

static struct starpu_perfmodel nl_energy_model =
{
	.type = STARPU_NL_REGRESSION_BASED,
	.symbol = "non_linear_memset_regression_based_energy"
};

static struct starpu_codelet memset_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {memset_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {memset_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs = {memset0_cpu, memset_cpu},
	.cpu_funcs_name = {"memset0_cpu", "memset_cpu"},
	.model = &model,
	.energy_model = &energy_model,
	.nbuffers = 1,
	.modes = {STARPU_SCRATCH}
};

static struct starpu_codelet nl_memset_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {memset_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {memset_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs = {memset0_cpu, memset_cpu},
	.cpu_funcs_name = {"memset0_cpu", "memset_cpu"},
	.model = &nl_model,
	.energy_model = &nl_energy_model,
	.nbuffers = 1,
	.modes = {STARPU_SCRATCH}
};

static void test_memset(int nelems, struct starpu_codelet *codelet)
{
	int nloops = 100;
	int loop;
	starpu_data_handle_t handle;

	starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, nelems, sizeof(int));
	for (loop = 0; loop < nloops; loop++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = codelet;
		task->handles[0] = handle;

		int ret = starpu_task_submit(task);
		if (ret == -ENODEV)
			exit(STARPU_TEST_SKIPPED);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_do_schedule();
	starpu_data_unregister(handle);
}

static int test_memset_energy(int nelems, int workerid, int where, enum starpu_worker_archtype archtype, int impl, struct starpu_codelet *codelet)
{
	(void)impl;
	int nloops;
	int loop;

	nloops = NENERGY;
	if (workerid == -1)
		nloops *= starpu_worker_get_count_by_type(archtype);

	starpu_data_handle_t handle[nloops];
	for (loop = 0; loop < nloops; loop++)
	{
		struct starpu_task *task = starpu_task_create();
		starpu_vector_data_register(&handle[loop], -1, (uintptr_t)NULL, nelems, sizeof(int));

		task->cl = codelet;
		task->where = where;
		task->handles[0] = handle[loop];
		task->flops = nelems;
		if (workerid != -1)
		{
			task->execute_on_a_specific_worker = 1;
			task->workerid = workerid;
		}

		int ret = starpu_task_submit(task);
		if (ret == -ENODEV)
			exit(STARPU_TEST_SKIPPED);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_do_schedule();
	for (loop = 0; loop < nloops; loop++)
	{
		starpu_data_unregister(handle[loop]);
	}

	return nloops;
}

static int bench_energy(int workerid, int where, enum starpu_worker_archtype archtype, int impl, struct starpu_codelet *codelet)
{
	int size;
	int retval;
	int ntasks;

	for (size = STARTlin; size < END; size *= 2)
	{
		starpu_data_handle_t handle;
		starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, size, sizeof(int));

		if ((retval = starpu_energy_start(workerid, archtype)) != 0)
		{
			starpu_data_unregister(handle);
			_STARPU_DISP("Energy measurement not supported for archtype %s\n", starpu_perfmodel_get_archtype_name(archtype));
			return -1;
		}

		/* Use a linear regression */
		ntasks = test_memset_energy(size, workerid, where, archtype, impl, codelet);

		struct starpu_task *task = starpu_task_create();
		task->cl = codelet;
		task->handles[0] = handle;
		task->synchronous = 1;
		task->destroy = 0;
		task->flops = size;

		retval = starpu_energy_stop(codelet->energy_model, task, impl, ntasks, workerid, archtype);

		starpu_task_destroy (task);
		starpu_data_unregister(handle);

		if (retval != 0)
			ERROR_RETURN(retval);
	}
	return 0;
}

static void show_task_perfs(int size, struct starpu_task *task)
{
	unsigned workerid;
	for (workerid = 0; workerid < starpu_worker_get_count(); workerid++)
	{
		char name[32];
		starpu_worker_get_name(workerid, name, sizeof(name));

		unsigned nimpl;
		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			FPRINTF(stdout, "Expected time for %d on %s (impl %u):\t%f\n",
				size, name, nimpl, starpu_task_expected_length(task, starpu_worker_get_perf_archtype(workerid, task->sched_ctx), nimpl));
		}
	}
}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(int argc, char **argv)
{
	struct starpu_conf conf;
	starpu_data_handle_t handle;
	int ret;
	unsigned i;

	starpu_conf_init(&conf);

	conf.sched_policy_name = "dmda";
	conf.calibrate = 2;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/perfmodels/opencl_memset_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	int size;
	for (size = STARTlin; size < END; size *= 2)
	{
		/* Use a linear regression */
		test_memset(size, &memset_cl);
	}

	for (size = START; size < END; size *= 2)
	{
		/* Use a non-linear regression */
		test_memset(size, &nl_memset_cl);
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

	/* Now create a dummy task just to estimate its duration according to the regression */

	size = 12345;

	starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, size, sizeof(int));

	struct starpu_task *task = starpu_task_create();
	task->cl = &memset_cl;
	task->handles[0] = handle;
	task->destroy = 0;

	show_task_perfs(size, task);

	task->cl = &nl_memset_cl;

	show_task_perfs(size, task);

	starpu_task_destroy(task);

	starpu_data_unregister(handle);

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
	starpu_shutdown();

	starpu_conf_init(&conf);

	/* Use a scheduler which doesn't choose the implementation */
#ifdef STARPU_HAVE_UNSETENV
	unsetenv("STARPU_SCHED");
#endif
	conf.sched_policy_name = "eager";
	conf.calibrate = 1;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/perfmodels/opencl_memset_kernel.cl",
						  &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	if (starpu_cpu_worker_get_count() > 0)
	{
		memset_cl.cpu_funcs[1] = NULL;
		bench_energy(-1, STARPU_CPU, STARPU_CPU_WORKER, 0, &memset_cl);
#ifdef STARPU_HAVE_UNSETENV
		memset_cl.cpu_funcs[1] = memset_cpu;
		memset_cl.cpu_funcs[0] = NULL;
		bench_energy(-1, STARPU_CPU, STARPU_CPU_WORKER, 1, &memset_cl);
#endif

		nl_memset_cl.cpu_funcs[1] = NULL;
		bench_energy(-1, STARPU_CPU, STARPU_CPU_WORKER, 0, &nl_memset_cl);
#ifdef STARPU_HAVE_UNSETENV
		nl_memset_cl.cpu_funcs[1] = memset_cpu;
		nl_memset_cl.cpu_funcs[0] = NULL;
		bench_energy(-1, STARPU_CPU, STARPU_CPU_WORKER, 1, &nl_memset_cl);
#endif
	}

	for (i = 0; i < starpu_cuda_worker_get_count(); i++)
	{
		int workerid = starpu_worker_get_by_type(STARPU_CUDA_WORKER, i);
		bench_energy(workerid, STARPU_CUDA, STARPU_CUDA_WORKER, 0, &memset_cl);
		bench_energy(workerid, STARPU_CUDA, STARPU_CUDA_WORKER, 0, &nl_memset_cl);
	}

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
	starpu_shutdown();

	return EXIT_SUCCESS;
}
