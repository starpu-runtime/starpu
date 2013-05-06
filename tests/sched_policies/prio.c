/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013 Université Bordeaux 1
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
#include <starpu_scheduler.h>
#include "../helper.h"

#ifdef STARPU_QUICK_CHECK
#define NTASKS 10
#else
#define NTASKS 1000
#endif

/*
 * Task1 must be executed before task0, even if task0 is submitted first.
 * Applies to : all schedulers.
 */

static void
A(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	printf("A");
	usleep(1000);
}

static void
B(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	printf("B");
	usleep(1000);
}

static int
run(struct starpu_sched_policy *policy)
{
	int ret;
	struct starpu_conf conf;
	int i;

	starpu_conf_init(&conf);
	conf.sched_policy = policy;
	ret = starpu_init(&conf);
	if (ret != 0)
		exit(STARPU_TEST_SKIPPED);
	starpu_profiling_status_set(1);

	struct starpu_codelet clA =
	{
		.cpu_funcs = {A, NULL},
		.nbuffers = 0
	};

	struct starpu_codelet clB =
	{
		.cpu_funcs = {B, NULL},
		.nbuffers = 0
	};

	for (i = 0; i < NTASKS; i++) {
		struct starpu_task *task = starpu_task_create();

		if (i%2) {
			task->cl = &clA;
			task->priority=STARPU_MIN_PRIO;
		} else {
			task->cl = &clB;
			task->priority=STARPU_MAX_PRIO;
		}
		task->detach=1;
		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();
	printf("\n");

	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
	return -ENODEV;
}

int
main(void)
{
	struct starpu_sched_policy **policies;
	struct starpu_sched_policy **policy;

	policies = starpu_sched_get_predefined_policies();
	for(policy=policies ; *policy!=NULL ; policy++)
	{
		int ret;

		FPRINTF(stderr, "Running with policy %s.\n", (*policy)->policy_name);
		ret = run(*policy);
		if (ret == -ENODEV)
			return STARPU_TEST_SKIPPED;
		if (ret == 1)
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
