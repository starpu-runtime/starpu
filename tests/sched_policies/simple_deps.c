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

#include <unistd.h>
#include <starpu.h>
#include <starpu_scheduler.h>
#include "../helper.h"

/*
 * Task1 must be executed before task0, even if task0 is submitted first.
 * Applies to : all schedulers.
 */

void dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	usleep(10000);
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
		exit(STARPU_TEST_SKIPPED);
	starpu_profiling_status_set(1);

	struct starpu_codelet cl =
	{
		.cpu_funcs = {dummy},
		.cpu_funcs_name = {"dummy"},
		.opencl_funcs = {dummy},
		.cuda_funcs = {dummy},
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
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(task1);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();

	double task1_end, task0_start;
	task1_end   = starpu_timing_timespec_to_us(&task1->profiling_info->end_time);
	task0_start = starpu_timing_timespec_to_us(&task0->profiling_info->start_time);

	starpu_task_destroy(task0);
	starpu_task_destroy(task1);
	starpu_shutdown();

	return !!(task1_end > task0_start);

enodev:
	starpu_shutdown();
	return -ENODEV;
}

int main(void)
{
	struct starpu_sched_policy **policies;
	struct starpu_sched_policy **policy;

	char *sched = getenv("STARPU_SCHED");

	policies = starpu_sched_get_predefined_policies();
	for(policy=policies ; *policy!=NULL ; policy++)
	{
		int ret;

		if (sched && strcmp(sched, (*policy)->policy_name))
			/* Testing another specific scheduler, no need to run this */
			continue;

		FPRINTF(stderr, "Running with policy %s.\n", (*policy)->policy_name);
		ret = run(*policy);
		if (ret == -ENODEV)
			return STARPU_TEST_SKIPPED;
		if (ret == 1)
			return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
