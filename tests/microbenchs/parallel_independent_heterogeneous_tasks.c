/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2016       Bérangère Subervie
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

#include <stdbool.h>
#include <starpu.h>
#include "../helper.h"

/* Run a series of independent tasks with heterogeneous execution time */

#define TIME 0.010
#ifdef STARPU_QUICK_CHECK
#define TASK_COEFFICIENT 20
#define MARGIN 0.20
#else
#define TASK_COEFFICIENT 100
#define MARGIN 0.10
#endif
#define TIME_CUDA_COEFFICIENT 10
#define TIME_OPENCL_COEFFICIENT 5
#define SECONDS_SCALE_COEFFICIENT_TIMING_NOW 1000000

void wait_CPU(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
	starpu_sleep(TIME);
}

void wait_CUDA(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
	starpu_sleep(TIME/TIME_CUDA_COEFFICIENT);
}

void wait_OPENCL(void *descr[], void *_args)
{
	(void)descr;
	(void)_args;
	starpu_sleep(TIME/TIME_OPENCL_COEFFICIENT);
}

double cost_function(struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	(void) t; (void) i;
	STARPU_ASSERT(a->ndevices == 1);
	if (a->devices[0].type == STARPU_CPU_WORKER)
	{
		STARPU_ASSERT(a->devices[0].ncores == 1);
		return TIME * 1000000;
	}
	else if (a->devices[0].type == STARPU_CUDA_WORKER)
	{
		return TIME/TIME_CUDA_COEFFICIENT * 1000000;
	}
	else if (a->devices[0].type == STARPU_OPENCL_WORKER)
	{
		return TIME/TIME_OPENCL_COEFFICIENT * 1000000;
	}
	STARPU_ASSERT(0);
	return 0.0;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
};

static struct starpu_codelet cl =
{
	.cpu_funcs = { wait_CPU },
	.cuda_funcs = { wait_CUDA },
	.opencl_funcs = { wait_OPENCL },
	.cpu_funcs_name = { "wait_CPU" },
	.nbuffers = 0,
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
	.model = &perf_model,
};

int main(int argc, char *argv[])
{
	int ret;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned nb_tasks, nb_workers_CPU, nb_workers_CUDA, nb_workers_OPENCL, i;
	double begin_time, end_time, time_m, time_s, speed_up, expected_speed_up, percentage_expected_speed_up;
	bool check, check_sup;

	nb_workers_CPU = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
	nb_workers_CUDA = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
	nb_workers_OPENCL = starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER);
	nb_tasks = (nb_workers_CPU + nb_workers_CUDA + nb_workers_OPENCL)*TASK_COEFFICIENT;

	begin_time = starpu_timing_now();

	/*execution des tasks*/

	for (i=0; i<nb_tasks; i++)
	{
		starpu_task_insert(&cl,0);
	}

	starpu_task_wait_for_all();

	end_time = starpu_timing_now();

	/*on determine si le temps mesure est satisfaisant ou pas*/

	time_m = (end_time - begin_time)/SECONDS_SCALE_COEFFICIENT_TIMING_NOW; //pour ramener en secondes
	time_s = nb_tasks * TIME;
	speed_up = time_s/time_m;
	expected_speed_up = nb_workers_CPU + TIME_CUDA_COEFFICIENT*nb_workers_CUDA + TIME_OPENCL_COEFFICIENT*nb_workers_OPENCL;
	percentage_expected_speed_up = 100 * (speed_up/expected_speed_up);
	check = speed_up >= (1 - MARGIN) * expected_speed_up;
	check_sup = speed_up <= (1 + MARGIN) * expected_speed_up;

	printf("measured time = %f seconds\nsequential time = %f seconds\nspeed up = %f\nnumber of workers CPU = %u\nnumber of workers CUDA = %u\nnumber of workers OPENCL = %u\nnumber of tasks = %u\nexpected speed up = %f\npercentage of expected speed up = %.2f%%\n", time_m, time_s, speed_up, nb_workers_CPU, nb_workers_CUDA, nb_workers_OPENCL, nb_tasks, expected_speed_up, percentage_expected_speed_up);

	starpu_shutdown();

	//test reussi ou test echoue
	if (check && check_sup)
	{
		return EXIT_SUCCESS;
	}
	else
	{
		return EXIT_FAILURE;
	}
}
