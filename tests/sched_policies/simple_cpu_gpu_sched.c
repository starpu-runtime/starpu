/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#include <core/perfmodel/perfmodel.h>

/*
 * Schedulers that are aware of the expected task length provided by the
 * perfmodels must make sure that :
 * 	- cpu_task is cheduled on a CPU.
 * 	- gpu_task is scheduled on a GPU.
 *
 * Applies to : dmda and to what other schedulers ?
 */

void dummy(void *buffers[], void *args)
{
	(void) buffers;
	(void) args;
}

/*
 * Fake cost functions.
 */
static double
cpu_task_cpu(struct starpu_task *task,
	     struct starpu_perfmodel_arch* arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;
	return 1.0;
}

static double
cpu_task_gpu(struct starpu_task *task,
	     struct starpu_perfmodel_arch* arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;

	return 10000000.0;
}

static double
gpu_task_cpu(struct starpu_task *task,
	     struct starpu_perfmodel_arch* arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;

	return 10000000.0;
}

static double
gpu_task_gpu(struct starpu_task *task,
	     struct starpu_perfmodel_arch* arch,
	     unsigned nimpl)
{
	(void) task;
	(void) arch;
	(void) nimpl;

	return 1.0;
}

static struct starpu_perfmodel model_cpu_task =
{
	.type = STARPU_PER_ARCH,
	.symbol = "model_cpu_task"
};

static struct starpu_perfmodel model_gpu_task =
{
	.type = STARPU_PER_ARCH,
	.symbol = "model_gpu_task"
};

static void
init_perfmodels_gpu(int gpu_type)
{
	int nb_worker_gpu = starpu_worker_get_count_by_type(gpu_type);
	int *worker_gpu_ids = malloc(nb_worker_gpu * sizeof(int));
	int worker_gpu;

	starpu_worker_get_ids_by_type(gpu_type, worker_gpu_ids, nb_worker_gpu);
	for(worker_gpu = 0 ; worker_gpu < nb_worker_gpu ; worker_gpu ++)
	{
		starpu_perfmodel_set_per_devices_cost_function(&model_cpu_task, 0, cpu_task_gpu,
							       gpu_type, starpu_worker_get_devid(worker_gpu_ids[worker_gpu]), 1,
							       -1);

		starpu_perfmodel_set_per_devices_cost_function(&model_gpu_task, 0, gpu_task_gpu,
							       gpu_type, starpu_worker_get_devid(worker_gpu_ids[worker_gpu]), 1,
							       -1);
	}
	free(worker_gpu_ids);
}

static void
init_perfmodels(void)
{
	starpu_perfmodel_init(&model_cpu_task);
	starpu_perfmodel_init(&model_gpu_task);

	starpu_perfmodel_set_per_devices_cost_function(&model_cpu_task, 0, cpu_task_cpu, STARPU_CPU_WORKER, 0, 1, -1);
	starpu_perfmodel_set_per_devices_cost_function(&model_gpu_task, 0, gpu_task_cpu, STARPU_CPU_WORKER, 0, 1, -1);

	// We need to set the cost function for each combination with a CUDA or a OpenCL worker
	init_perfmodels_gpu(STARPU_CUDA_WORKER);
	init_perfmodels_gpu(STARPU_OPENCL_WORKER);
}

/*
 * Dummy codelets.
 */
static struct starpu_codelet cpu_cl =
{
	.cpu_funcs    = { dummy },
	.cuda_funcs   = { dummy },
	.opencl_funcs = { dummy },
	.nbuffers     = 0,
	.model        = &model_cpu_task
};

static struct starpu_codelet gpu_cl =
{
	.cpu_funcs    = { dummy },
	.cuda_funcs   = { dummy },
	.opencl_funcs = { dummy },
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
	{
		starpu_shutdown();
		exit(STARPU_TEST_SKIPPED);
	}
	if (starpu_cuda_worker_get_count() == 0 && starpu_opencl_worker_get_count() == 0)
	{
		starpu_shutdown();
		exit(STARPU_TEST_SKIPPED);
	}

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

	enum starpu_worker_archtype cpu_task_worker, gpu_task_worker;
	cpu_task_worker = starpu_worker_get_type(cpu_task->profiling_info->workerid);
	gpu_task_worker = starpu_worker_get_type(gpu_task->profiling_info->workerid);
	if (cpu_task_worker != STARPU_CPU_WORKER || (gpu_task_worker != STARPU_CUDA_WORKER && gpu_task_worker != STARPU_OPENCL_WORKER))
	{
		FPRINTF(stderr, "Tasks did not execute on expected worker\n");
		if (cpu_task_worker != STARPU_CPU_WORKER)
		{
			FPRINTF(stderr, "The CPU task did not run on a CPU worker\n");
		}
		if (gpu_task_worker != STARPU_CUDA_WORKER && gpu_task_worker != STARPU_OPENCL_WORKER)
		{
			FPRINTF(stderr, "The GPU task did not run on a Cuda or OpenCL worker\n");
		}

		ret = 1;
	}
	else
	{
		FPRINTF(stderr, "Tasks DID execute on expected worker\n");
		ret = 0;
	}

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
extern struct starpu_sched_policy _starpu_sched_dmda_ready_policy;
extern struct starpu_sched_policy _starpu_sched_dmda_sorted_policy;
extern struct starpu_sched_policy _starpu_sched_eager_policy;
extern struct starpu_sched_policy _starpu_sched_parallel_heft_policy;
extern struct starpu_sched_policy _starpu_sched_peager_policy;
*/
extern struct starpu_sched_policy _starpu_sched_dmda_policy;

/* XXX: what policies are we interested in ? */
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
	//&_starpu_sched_parallel_heft_policy,
	//&_starpu_sched_peager_policy
};

int main(void)
{
#ifndef STARPU_HAVE_SETENV
/* XXX: is this macro used by all the schedulers we are interested in ? */
#warning "setenv() is not available, skipping this test"
	return STARPU_TEST_SKIPPED;
#else
	setenv("STARPU_SCHED_BETA", "0", 1);

	char *sched = getenv("STARPU_SCHED");

	if (starpu_get_env_number_default("STARPU_NWORKER_PER_CUDA", 1) != 1)
		return STARPU_TEST_SKIPPED;

	int i;
	int n_policies = sizeof(policies)/sizeof(policies[0]);
	for (i = 0; i < n_policies; ++i)
	{
		struct starpu_sched_policy *policy = policies[i];

		if (sched && strcmp(sched, policy->policy_name))
			/* Testing another specific scheduler, no need to run this */
			continue;

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
