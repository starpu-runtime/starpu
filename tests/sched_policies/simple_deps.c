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

#include <unistd.h>

#include <starpu.h>
#include <starpu_profiling.h>


#include "../helper.h"

/*
 * Task1 must be executed before task0, even if task0 is submitted first.
 * Applies to : all schedulers.
 */

static void
dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	sleep(1);
}

static int
run(struct starpu_sched_policy *policy)
{
	int ret;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.sched_policy = policy;
	ret = starpu_init(&conf);
	if (ret != 0)
		return 1;
	starpu_profiling_status_set(1);

	struct starpu_codelet cl =
	{
		.cpu_funcs = {dummy, NULL},
		.nbuffers = 0
	};

	struct starpu_task *task0 = starpu_task_create();
	task0->cl = &cl;
	task0->destroy = 0;
	
	struct starpu_task *task1 = starpu_task_create();
	task1->cl = &cl;
	task1->destroy = 0;

	starpu_task_declare_deps_array(task0, 1, &task1);

	ret = starpu_task_submit(task0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(task1);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();

	double t1, t2;
	t1 = starpu_timing_timespec_to_us(&task1->profiling_info->end_time);
	t2 = starpu_timing_timespec_to_us(&task0->profiling_info->start_time);

	starpu_task_destroy(task0);
	starpu_task_destroy(task1);
	starpu_shutdown();	

	return t1 < t2 ? 0:1;
}

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
