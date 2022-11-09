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

/*
 * A multi-implementation benchmark with dmda scheduler
 * we aim to test OPENCL workers and calculate the estimated time for each type of worker (CPU or OPENCL or CUDA)
 * dmda choose OPENCL workers for lage size (variable size of compare_performance) size=1234567
 * dmda choose CPU workers for small size (size=1234)
 */

#define STARTlin 131072
#define START 1024
#ifdef STARPU_QUICK_CHECK
#define END 1048576
#else
#define END 16777216
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
extern void memset0_opencl(void *buffers[], void *args);
extern void memset_opencl(void *buffers[], void *args);
#endif

void memset0_cpu(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
	unsigned i;

	//starpu_usleep(100);

	for (i = 0; i < n; i++)

		ptr[0] += i;
}

void memset_cpu(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);

	//starpu_usleep(10);
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

static struct starpu_codelet memset_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {memset_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {memset0_opencl, memset_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs = {memset0_cpu, memset_cpu},
	.cpu_funcs_name = {"memset0_cpu", "memset_cpu"},
	.model = &model,
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
	.opencl_funcs = {memset0_opencl, memset_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs = {memset0_cpu, memset_cpu},
	.cpu_funcs_name = {"memset0_cpu", "memset_cpu"},
	.model = &nl_model,
	.nbuffers = 1,
	.modes = {STARPU_SCRATCH}
};

static void test_memset(int nelems, struct starpu_codelet *codelet)
{
#ifdef STARPU_QUICK_CHECK
	int nloops = 10;
#else
	int nloops = 100;
#endif
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

static void compare_performance(int size, struct starpu_codelet *codelet, struct starpu_task *compar_task)
{
	unsigned i;
#ifdef STARPU_QUICK_CHECK
	unsigned niter = 10;
#else
	unsigned niter = 100;
#endif
	starpu_data_handle_t handle;

	starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, size, sizeof(int));

	struct starpu_task *tasks[niter];

	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = codelet;
		task->handles[0] = handle;

		task->synchronous = 1;

		/* We will destroy the task structure by hand so that we can
		 * query the profiling info before the task is destroyed. */
		task->destroy = 0;

		tasks[i] = task;

		int ret = starpu_task_submit(task);

		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			FPRINTF(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_data_unregister(handle);

	starpu_task_wait_for_all();

	double length_cpu_sum = 0.0;
	double length_gpu_sum = 0.0;

	enum starpu_worker_archtype archi;

	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = tasks[i];
		struct starpu_profiling_task_info *info = task->profiling_info;

		//archi=starpu_worker_get_type(0);
		archi=starpu_worker_get_type(info->workerid);

		switch (archi)
		{
		case STARPU_CPU_WORKER:
			FPRINTF(stdout, "cpuuu\n");
			/* How long was the task execution ? */
			length_cpu_sum += starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);
			break;

		case STARPU_OPENCL_WORKER:

			FPRINTF(stdout, "openclllllll\n");
			/* How long was the task execution ? */
			length_gpu_sum += starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);
			break;

		case STARPU_CUDA_WORKER:

			FPRINTF(stdout, "cudaaaaaa\n");
			/* How long was the task execution ? */
			length_gpu_sum += starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);
			break;


	default:
			FPRINTF(stdout, "unsupported!\n");
		break;
		}

		/* We don't need the task structure anymore */
		starpu_task_destroy(task);

	}

	unsigned worker;

	/* Display the occupancy of all workers during the test */
	unsigned ncpus =  starpu_cpu_worker_get_count();
	unsigned ngpus =  starpu_opencl_worker_get_count()+starpu_cuda_worker_get_count();
	//unsigned ncpu= starpu_worker_get_count_by_type(STARPU_CPU_WORKER);

	FPRINTF(stderr, "ncpus %u \n", ncpus);
	FPRINTF(stderr, "ngpus %u \n", ngpus);
	for (worker= 0; worker< starpu_worker_get_count(); worker++)
	{

		struct starpu_profiling_worker_info worker_info;
		int ret = starpu_profiling_worker_get_info(worker, &worker_info);
		STARPU_ASSERT(!ret);

		char workername[128];
		starpu_worker_get_name(worker, workername, sizeof(workername));
		unsigned nimpl;

		FPRINTF(stdout, "\n Worker :%s ::::::::::\n\n", workername);

		for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			switch (starpu_worker_get_type(worker))

			{
			case STARPU_CPU_WORKER:

				FPRINTF(stdout, "Expected time for %d on %s (impl %u): %f, Measured time: %f \n",
						size, workername, nimpl,starpu_task_expected_length(compar_task, starpu_worker_get_perf_archtype(worker, compar_task->sched_ctx), nimpl), ((length_cpu_sum)/niter));

				break;

			case STARPU_OPENCL_WORKER:

				FPRINTF(stdout, "Expectedd time for %d on %s (impl %u): %f, Measuredd time: %f \n",
						size, workername, nimpl,starpu_task_expected_length(compar_task, starpu_worker_get_perf_archtype(worker, compar_task->sched_ctx), nimpl), ((length_gpu_sum)/niter));

				break;

			case STARPU_CUDA_WORKER:

				FPRINTF(stdout, "Expectedd time for %d on %s (impl %u): %f, Measuredd time: %f \n",
						size, workername, nimpl,starpu_task_expected_length(compar_task, starpu_worker_get_perf_archtype(worker, compar_task->sched_ctx), nimpl), ((length_gpu_sum)/niter));

				break;

			default:
				FPRINTF(stdout, "unsupported!\n");
				break;
			}
		}
	}


}

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

int main(int argc, char **argv)
{
	/* Enable profiling */
	starpu_profiling_status_set(STARPU_PROFILING_ENABLE);

	struct starpu_conf conf;
	starpu_data_handle_t handle;
	int ret;

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

	for (size = START*1.5; size < END; size *= 2)
	{
		/* Use a non-linear regression */
		test_memset(size, &nl_memset_cl);
	}

	ret = starpu_task_wait_for_all();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_wait_for_all");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
	starpu_shutdown();


	/* Test Phase */
	starpu_conf_init(&conf);

	conf.sched_policy_name = "dmda";
	conf.calibrate = 0;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("tests/perfmodels/opencl_memset_kernel.cl",
			&opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	/* Now create a dummy task just to estimate its duration according to the regression */

	size = 1234567;

	starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, size, sizeof(int));

	struct starpu_task *task = starpu_task_create();
	task->handles[0] = handle;
	task->destroy = 0;

	//FPRINTF(stdout, "\n ////linear regression results////\n");
	//task->cl = &memset_cl;
	//compare_performance(size, &memset_cl, task);

	FPRINTF(stdout, "\n ////non linear regression results////\n");
	task->cl = &nl_memset_cl;
	compare_performance(size, &nl_memset_cl, task);

	starpu_task_destroy(task);

	starpu_data_unregister(handle);

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
	starpu_shutdown();

	return EXIT_SUCCESS;
}
