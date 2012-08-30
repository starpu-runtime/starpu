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
#include <starpu_profiling.h>

#include "../helper.h"

/*
 * Schedulers that are aware of the expected task length provided by the
 * perfmodels must make sure that :
 * 	- cpu_task is cheduled on a CPU.
 * 	- gpu_task is scheduled on a GPU.
 *
 * Applies to : heft, XXX : and to what other schedulers ?
 */


static void
dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
}

/*
 * Fake cost functions.
 */
static double
cpu_task_cpu(struct starpu_task *task,
	     enum starpu_perf_archtype arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;
	return 1.0;
}

static double
cpu_task_gpu(struct starpu_task *task,
	     enum starpu_perf_archtype arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;

	return 1000.0;
}

static double
gpu_task_cpu(struct starpu_task *task,
	     enum starpu_perf_archtype arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;

	return 1000.0;
}

static double
gpu_task_gpu(struct starpu_task *task,
	     enum starpu_perf_archtype arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;

	return 1.0;
}

static struct starpu_perfmodel model_cpu_task = 
{
	.type = STARPU_PER_ARCH
};
static struct starpu_perfmodel model_gpu_task = 
{
	.type = STARPU_PER_ARCH
};

static void
init_perfmodels(void)
{
	int i;
	for (i = STARPU_CPU_DEFAULT; i < STARPU_CUDA_DEFAULT; i++)
	{
		model_cpu_task.per_arch[i][0].cost_function = cpu_task_cpu;
		model_gpu_task.per_arch[i][0].cost_function = gpu_task_cpu;
	}
	for (i = STARPU_CUDA_DEFAULT; i < STARPU_GORDON_DEFAULT; i++)
	{
		model_cpu_task.per_arch[i][0].cost_function = cpu_task_gpu;
		model_gpu_task.per_arch[i][0].cost_function = gpu_task_gpu;
	}
}

/*
 * Dummy codelets.
 */
static struct starpu_codelet cpu_cl =
{
	.cpu_funcs    = { dummy, NULL },
	.cuda_funcs   = { dummy, NULL },
	.opencl_funcs = { dummy, NULL },
	.nbuffers     = 0,
	.model        = &model_cpu_task
};

static struct starpu_codelet gpu_cl =
{
	.cpu_funcs    = { dummy, NULL },
	.cuda_funcs   = { dummy, NULL },
	.opencl_funcs = { dummy, NULL },
	.nbuffers     = 0,
	.model        = &model_gpu_task
};

static int
run(struct starpu_sched_policy *policy)
{
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.sched_policy = policy;
	int ret = starpu_init(&conf);
	if (ret == -ENODEV)
		exit(STARPU_TEST_SKIPPED);

	/* At least 1 CPU and 1 GPU are needed. */
	if (starpu_cpu_worker_get_count() == 0)
		exit(STARPU_TEST_SKIPPED);
	if (starpu_cuda_worker_get_count() == 0 &&
	    starpu_opencl_worker_get_count() == 0)
		exit(STARPU_TEST_SKIPPED);

	starpu_profiling_status_set(1);
	init_perfmodels();

	struct starpu_task *cpu_task = starpu_task_create();
	cpu_task->cl = &cpu_cl;
	cpu_task->destroy = 0;

	struct starpu_task *gpu_task = starpu_task_create();
	gpu_task->cl = &gpu_cl;
	gpu_task->destroy = 0;

	ret = starpu_task_submit(cpu_task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	ret = starpu_task_submit(gpu_task);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();

	enum starpu_archtype cpu_task_worker, gpu_task_worker;
	cpu_task_worker = starpu_worker_get_type(cpu_task->profiling_info->workerid);
	gpu_task_worker = starpu_worker_get_type(gpu_task->profiling_info->workerid);
	if (cpu_task_worker != STARPU_CPU_WORKER ||
	    (gpu_task_worker != STARPU_CUDA_WORKER &&
	     gpu_task_worker != STARPU_OPENCL_WORKER))
		ret = 1;
	else
		ret = 0;


	starpu_task_destroy(cpu_task);
	starpu_task_destroy(gpu_task);
	starpu_shutdown();
	return ret;
}

/*
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
*/
extern struct starpu_sched_policy _starpu_sched_heft_policy;

/* XXX: what policies are we interested in ? */
static struct starpu_sched_policy *policies[] =
{
	//&_starpu_sched_ws_policy,
	//&_starpu_sched_prio_policy,
	//&_starpu_sched_dm_policy,
	//&_starpu_sched_dmda_policy,
	&_starpu_sched_heft_policy,
	//&_starpu_sched_dmda_ready_policy,
	//&_starpu_sched_dmda_sorted_policy,
	//&_starpu_sched_random_policy,
	//&_starpu_sched_eager_policy,
	//&_starpu_sched_parallel_heft_policy,
	//&_starpu_sched_pgreedy_policy
};

int
main(void)
{
#ifndef STARPU_HAVE_SETENV
/* XXX: is this macro used by all the schedulers we are interested in ? */
#warning "setenv() is not available, skipping this test"
	return STARPU_TEST_SKIPPED;
#else
	setenv("STARPU_SCHED_BETA", "0", 1);

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
#endif
}
