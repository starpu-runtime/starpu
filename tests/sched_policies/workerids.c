/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013	    Simon Archipoff
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
 * Check that the starpu_task::workerids field is respected by schedulers.
 */

#ifdef STARPU_QUICK_CHECK
#define NTASKS 10
#elif !defined(STARPU_LONG_CHECK)
#define NTASKS 100
#else
#define NTASKS 1000
#endif

void funcA(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
	int id = starpu_worker_get_id();
	STARPU_ASSERT_MSG(id == 0, "Expected id 0 but got %d\n", id);
	starpu_usleep(1000);
}

double cost_function(struct starpu_task *t STARPU_ATTRIBUTE_UNUSED, struct starpu_perfmodel_arch *a STARPU_ATTRIBUTE_UNUSED, unsigned i STARPU_ATTRIBUTE_UNUSED)
{
	return 1000;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
};

static struct starpu_codelet clA =
{
	.cpu_funcs = {funcA},
	.cpu_funcs_name = {"funcA"},
	.opencl_funcs = {funcA},
	.cuda_funcs = {funcA},
	.hip_funcs = {funcA},
	.max_fpga_funcs = {funcA},
	.nbuffers = 0,
	.model = &perf_model,
};

static int run(struct starpu_sched_policy *policy)
{
	int ret;
	struct starpu_conf conf;
	int i;

	starpu_conf_init(&conf);
	conf.sched_policy = policy;
	ret = starpu_init(&conf);
	if (ret != 0)
		exit(STARPU_TEST_SKIPPED);

	uint32_t zeromask = 1;
	for (i = 0; i < NTASKS; i++)
	{
		struct starpu_task *task = starpu_task_create();

		task->cl = &clA;
		task->workerids = &zeromask;
		task->workerids_len = 1;
		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();
	FPRINTF(stdout,"\n");

	starpu_shutdown();
	return 0;

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

		if (strcmp((*policy)->policy_name, "lws") == 0
		 || strcmp((*policy)->policy_name, "ws") == 0)
#ifdef STARPU_DEVEL
#warning FIXME performance for ws
#endif
			continue;

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
