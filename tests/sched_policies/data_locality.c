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
#include <starpu_scheduler.h>

#include "../helper.h"

/*
 * Check that scheduling policies tend to put tasks on the worker which has a
 * copy of the data
 */

#define NTASKS 8

/*
 * It is very inefficient to keep moving data between memory nodes. This
 * test makes sure the scheduler will take account of the data locality
 * when scheduling tasks.
 *
 * Applies to : dmda, pheft.
 */

void dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
}

/*
 * Dummy cost function, used to make sure the scheduler does schedule the
 * task, instead of getting rid of it as soon as possible because it doesn't
 * know its expected length.
 */
static double
cost_function(struct starpu_task *task, unsigned nimpl)
{
	(void) task;
	(void) nimpl;
	return 1.0;
}

static struct starpu_perfmodel model =
{
	.type          = STARPU_COMMON,
	.cost_function = cost_function
};

static struct starpu_codelet cl =
{
	.cpu_funcs     = { dummy },
	.cuda_funcs    = { dummy },
	.opencl_funcs  = { dummy },
	.modes         = { STARPU_RW },
	.model         = &model,
	.nbuffers      = 1
};

static int var = 42;
static starpu_data_handle_t rw_handle;

static void
init_data(void)
{
	starpu_variable_data_register(&rw_handle, STARPU_MAIN_RAM, (uintptr_t) &var,
					sizeof(var));
}

static void
free_data(void)
{
	starpu_data_unregister(rw_handle);
}

static int
run(struct starpu_sched_policy *policy)
{
	int ret;
	struct starpu_conf conf;

	starpu_conf_init(&conf);
	conf.sched_policy = policy;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
	{
		FPRINTF(stderr, "No device found\n");
		return -ENODEV;
	}

	if (starpu_cpu_worker_get_count() == 0 || (starpu_cuda_worker_get_count() == 0 && starpu_opencl_worker_get_count() == 0))
		goto enodev;

	starpu_profiling_status_set(1);
	init_data();

	/* Send the handle to a GPU. */
	cl.where = STARPU_CUDA | STARPU_OPENCL;
	struct starpu_task *tasks[NTASKS];
	tasks[0] = starpu_task_create();
	tasks[0]->cl = &cl;
	tasks[0]->synchronous = 1;
	tasks[0]->handles[0] = rw_handle;
	tasks[0]->destroy = 0;
	ret = starpu_task_submit(tasks[0]);
	if (ret == -ENODEV)
		goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");


	/* Now, run multiple tasks using this handle. */
	cl.where |= STARPU_CPU;
	int i;
	for (i = 1; i < NTASKS; i++)
	{
		tasks[i] = starpu_task_create();
		tasks[i]->cl = &cl;
		tasks[i]->handles[0] = rw_handle;
		tasks[i]->destroy = 0;
		ret = starpu_task_submit(tasks[i]);
		if (ret == -ENODEV)
			goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	starpu_task_wait_for_all();

	/* All tasks should have been executed on the same GPU. */
	ret = 0;
	int workerid = tasks[0]->profiling_info->workerid;
	for (i = 0; i < NTASKS; i++)
	{
		if (tasks[i]->profiling_info->workerid != workerid)
		{
			FPRINTF(stderr, "Error for task %d. Worker id %d different from expected worker id %d\n", i, tasks[i]->profiling_info->workerid, workerid);
			ret = 1;
			break;
		}
		starpu_task_destroy(tasks[i]);
	}

	/* Clean everything up. */
	for (; i < NTASKS; i++)
		starpu_task_destroy(tasks[i]);

	free_data();
	starpu_shutdown();

	return ret;

enodev:
	FPRINTF(stderr, "No device found\n");
	starpu_shutdown();
	return -ENODEV;
}

/* XXX: Does this test apply to other schedulers ? */
//extern struct starpu_sched_policy _starpu_sched_ws_policy;
//extern struct starpu_sched_policy _starpu_sched_prio_policy;
//extern struct starpu_sched_policy _starpu_sched_random_policy;
//extern struct starpu_sched_policy _starpu_sched_dm_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_policy;
//extern struct starpu_sched_policy _starpu_sched_dmda_ready_policy;
//extern struct starpu_sched_policy _starpu_sched_dmda_sorted_policy;
//extern struct starpu_sched_policy _starpu_sched_eager_policy;
extern struct starpu_sched_policy _starpu_sched_parallel_heft_policy;
//extern struct starpu_sched_policy _starpu_sched_peager_policy;

static struct starpu_sched_policy *policies[] =
{
	//&_starpu_sched_ws_policy,
	//&_starpu_sched_prio_policy,
	//&_starpu_sched_dm_policy,
	&_starpu_sched_dmda_policy,
	//&_starpu_sched_dmda_ready_policy,
	//&_starpu_sched_dmda_sorted_policy,
	//&_starpu_sched_random_policy,
	//&_starpu_sched_eager_policy,
	&_starpu_sched_parallel_heft_policy,
	//&_starpu_sched_peager_policy
};

int main(void)
{
	int i;
	int n_policies = sizeof(policies)/sizeof(policies[0]);
	int global_ret = 0;

	char *sched = getenv("STARPU_SCHED");

	for (i = 0; i < n_policies; ++i)
	{
		struct starpu_sched_policy *policy = policies[i];

		if (sched && strcmp(sched, policy->policy_name))
			/* Testing another specific scheduler, no need to run this */
			continue;

		FPRINTF(stdout, "Running with policy %s.\n", policy->policy_name);
		int ret = run(policy);
		if (ret == -ENODEV && global_ret == 0)
			global_ret = STARPU_TEST_SKIPPED;
		if (ret == 1 && global_ret == 0)
			global_ret = ret;
	}

	return global_ret;
}
