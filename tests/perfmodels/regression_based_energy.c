/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * we aim to test the energy model with the different size of gamma
 * for large size of gamma, dmda choose the second implementation which consumes less energy
 * otherwise, it choose the first implementation which minimizes the execution time
 */

#define STARTlin 131072
#define START 1024
#ifdef STARPU_QUICK_CHECK
#define END 1048576
#else
#define END 16777216
#endif

// first implementation with an initial delay (100 us)
void memset0_cpu(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
	unsigned i;

	starpu_usleep(100);

	for (i=0; i<n ; i++)
	{
		ptr[0] += i;
	}
}

// second implementation without initial delay but 2.5 more loops
void memset_cpu(void *descr[], void *arg)
{
	(void)arg;
	STARPU_SKIP_IF_VALGRIND;

	unsigned *ptr = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);
	unsigned n = STARPU_VECTOR_GET_NX(descr[0]);
	int i;

	for (i=0; i<6.5*n ; i++)
	{
		ptr[0] += i;
	}
}

// energy measurement function
double energy_function(struct starpu_task *task, struct starpu_perfmodel_arch *arch, unsigned nimpl)

{
	double energy;
	int factor;
	if (nimpl == 0)
		factor = 10;
	else
		factor = 1;

	energy=starpu_task_expected_length(task, arch, nimpl)*factor;

	return energy;
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

static struct starpu_perfmodel nl_energy_model=
{
	.type = STARPU_PER_ARCH,
	.symbol = "non_linear_energy_model",
	.arch_cost_function=energy_function,
};

static struct starpu_codelet memset_cl =
{
	.cpu_funcs = {memset0_cpu, memset_cpu},
	.cpu_funcs_name = {"memset0_cpu", "memset_cpu"},
	.model = &model,
	.nbuffers = 1,
	.modes = {STARPU_SCRATCH}
};

static struct starpu_codelet nl_memset_cl =
{
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

static void compare_performance(int size, struct starpu_codelet *codelet, struct starpu_task *compar_task)
{
	unsigned i;
	unsigned niter = 100;
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

	double length_sum = 0.0;

	for (i = 0; i < niter; i++)
	{
		struct starpu_task *task = tasks[i];
		struct starpu_profiling_task_info *info = task->profiling_info;


		/* How long was the task execution ? */
		length_sum += starpu_timing_timespec_delay_us(&info->start_time, &info->end_time);

		/* We don't need the task structure anymore */
		starpu_task_destroy(task);
	}


	/* Display the occupancy of all workers during the test */
	unsigned worker;
	for (worker = 0; worker < starpu_worker_get_count(); worker++)
	{
		struct starpu_profiling_worker_info worker_info;
		int ret = starpu_profiling_worker_get_info(worker, &worker_info);
		STARPU_ASSERT(!ret);

		char workername[128];
		starpu_worker_get_name(worker, workername, sizeof(workername));
		unsigned nimpl;

		if (starpu_worker_get_type(worker)==STARPU_CPU_WORKER)
		{
			FPRINTF(stdout, "\n Worker :%s ::::::::::\n\n", workername);

			for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{

				FPRINTF(stdout, "Expected time for %d on %s (impl %u): %f, Measured time: %f, Expected energy: %f\n",
						size, workername, nimpl,starpu_task_expected_length(compar_task, starpu_worker_get_perf_archtype(worker, compar_task->sched_ctx), nimpl), ((length_sum)/niter),
						starpu_task_expected_energy(compar_task, starpu_worker_get_perf_archtype(worker, compar_task->sched_ctx), nimpl));
			}
		}
	}


}


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

	starpu_shutdown();


	/* Test Phase */
	starpu_conf_init(&conf);

	conf.sched_policy_name = "dmda";
	conf.calibrate = 0;

	ret = starpu_initialize(&conf, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Now create a dummy task just to estimate its duration according to the regression */

	size = 1234567;

	starpu_vector_data_register(&handle, -1, (uintptr_t)NULL, size, sizeof(int));

	struct starpu_task *task = starpu_task_create();
	task->handles[0] = handle;
	task->destroy = 0;

	FPRINTF(stdout, "\n ////non linear regression results////\n");
	task->cl = &nl_memset_cl;
	compare_performance(size, &nl_memset_cl, task);

	starpu_task_destroy(task);

	starpu_data_unregister(handle);

	starpu_shutdown();

	return EXIT_SUCCESS;
}
