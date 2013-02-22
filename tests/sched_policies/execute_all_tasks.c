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

#include <starpu.h>
#include <core/jobs.h>
#include "../helper.h"

/*
 * All tasks submitted by StarPU should be executed once.
 * Applies to: all schedulers.
 */

#define NTASKS           8

static void
dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
}

static int
run(struct starpu_sched_policy *p)
{
	int ret;
	struct starpu_conf conf;

	(void) starpu_conf_init(&conf);
	conf.sched_policy = p;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		exit(STARPU_TEST_SKIPPED);

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

	ret = 0;
	for (i = 0; i < NTASKS; i++)
	{
		struct _starpu_job *j = tasks[i]->starpu_private;
		if (j == NULL || j->terminated == 0)
		{
			FPRINTF(stderr, "Error with policy %s.\n", p->policy_name);
			ret = 1;
			break;
		}
	}

	for (i = 0; i < NTASKS; i++)
	{
		starpu_task_destroy(tasks[i]);
	}

	starpu_shutdown();
	return ret;
}

int
main(void)
{
	struct starpu_sched_policy **policies;
	struct starpu_sched_policy **policy;

	policies = starpu_sched_get_predefined_policies();
	for(policy=policies ; *policy!=NULL ; policy++)
	{
		FPRINTF(stderr, "Running with policy %s.\n", (*policy)->policy_name);
		int ret;
		ret = run(*policy);
		if (ret == 1)
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
