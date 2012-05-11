/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 Inria
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

#include <math.h>
#include <unistd.h>

#include <starpu.h>
#include <starpu_profiling.h>

#include "../helper.h"

/*
 * All tasks submitted by StarPU should be executed once.
 * Applies to: all schedulers.
 */

#define NTASKS  2

extern struct starpu_sched_policy _starpu_sched_ws_policy;
extern struct starpu_sched_policy _starpu_sched_prio_policy;
extern struct starpu_sched_policy _starpu_sched_random_policy;
extern struct starpu_sched_policy _starpu_sched_dm_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_ready_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_sorted_policy;
extern struct starpu_sched_policy _starpu_sched_eager_policy;
extern struct starpu_sched_policy _starpu_sched_parallel_heft_policy;
extern struct starpu_sched_policy _starpu_sched_pgreedy_policy;
extern struct starpu_sched_policy heft_policy;

static struct starpu_sched_policy *policies[] =
{
	&_starpu_sched_ws_policy,
	&_starpu_sched_prio_policy,
	&_starpu_sched_dm_policy,
	&_starpu_sched_dmda_policy,
	&heft_policy,
	&_starpu_sched_dmda_ready_policy,
	&_starpu_sched_dmda_sorted_policy,
	&_starpu_sched_random_policy,
	&_starpu_sched_eager_policy,
	&_starpu_sched_parallel_heft_policy,
	&_starpu_sched_pgreedy_policy
};

static void
dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	usleep(1000000);
}

static int
run(struct starpu_sched_policy *p)
{
	int ret;
	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(STARPU_TEST_SKIPPED);

	starpu_profiling_status_set(1);

	struct starpu_task *tasks[NTASKS] = { NULL };
	struct starpu_codelet cl = 
	{
		.cpu_funcs    = {dummy, NULL},
		.cuda_funcs   = {dummy, NULL},
		.opencl_funcs = {dummy, NULL},
		.nbuffers     = 0
	};

	int i;
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task = starpu_task_create();
		tasks[i] = task;
		task->cl = &cl;
		task->synchronous = 1;
		task->destroy = 0;
		ret = starpu_task_submit(task);
		if (ret != 0)
			return 1;
	}

	starpu_task_wait_for_all();

	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task_profiling_info *pi;
		double task_len;

		pi = tasks[i]->profiling_info;
		task_len = starpu_timing_timespec_delay_us(&pi->start_time, &pi->end_time);
		if (fabs(task_len - 1e6) > 10000) /* That's 10ms, should be good. */
		{
			FPRINTF(stderr, "Failed with task length: %fµ\n", task_len);
			return 1;
		}

		starpu_task_destroy(tasks[i]);
	}

	starpu_shutdown();
	return 0;
}

int
main(void)
{
	int i;
	int n_policies = sizeof(policies)/sizeof(policies[0]);
	for (i = 0; i < n_policies; ++i)
	{
		struct starpu_sched_policy *policy = policies[i];
		FPRINTF(stdout, "Running with policy %s.\n",
			policy->policy_name);
		int ret;
		ret = run(policy);
		if (ret == 1)
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
