/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This is a dumb sample of stencil application
 *
 * Dumb domain split in N pieces:
 *
 * 0 | 1 | ... | N-1
 *
 * for each simulation iteration, a task works on some adjacent pieces
 *
 * Locality is thus set on the central piece.
 */

#include <starpu.h>
#include "../helper.h"

#define N 50

#define ITER 50

int task_worker[N][ITER];
int worker_task[STARPU_NMAXWORKERS][N*ITER];
unsigned worker_ntask[STARPU_NMAXWORKERS];

void cpu_f(void *descr[], void *_args)
{
	(void)descr;
	unsigned i, loop, worker = starpu_worker_get_id();
	enum starpu_worker_archtype worker_type = starpu_worker_get_type(worker);

	starpu_codelet_unpack_args(_args, &loop, &i);
	task_worker[i][loop] = worker;
	worker_task[worker][worker_ntask[worker]++] = i;
	if (worker_type == STARPU_CPU_WORKER)
		starpu_sleep(0.001);
	else
		starpu_sleep(0.0001);
}

double cost_function(struct starpu_task *t, struct starpu_perfmodel_arch *a, unsigned i)
{
	(void) t; (void) i;
	STARPU_ASSERT(a->ndevices == 1);
	if (a->devices[0].type == STARPU_CPU_WORKER)
	{
		STARPU_ASSERT(a->devices[0].ncores == 1);
		return 1000;
	}
	else
		return 100;
}

static struct starpu_perfmodel perf_model =
{
	.type = STARPU_PER_ARCH,
	.arch_cost_function = cost_function,
};

static struct starpu_codelet cl =
{
	.cpu_funcs = { cpu_f },
	.cpu_funcs_name = { "cpu_f" },
	.cuda_funcs = { cpu_f },
	.opencl_funcs = { cpu_f },
	.nbuffers = 4,
	.modes =
	{
		STARPU_RW,
		STARPU_RW | STARPU_COMMUTE | STARPU_LOCALITY,
		STARPU_RW | STARPU_COMMUTE | STARPU_LOCALITY,
		STARPU_RW | STARPU_COMMUTE | STARPU_LOCALITY,
	},
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
	.model = &perf_model,
};

int main(int argc, char *argv[])
{
	int ret;
	starpu_data_handle_t A[N];
	starpu_data_handle_t B[N];
	unsigned i, loop, finished;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Get most parallelism by using an arbiter */
	starpu_arbiter_t arbiter = starpu_arbiter_create();
	for (i = 0; i < N; i++)
	{
		starpu_void_data_register(&A[i]);
		starpu_void_data_register(&B[i]);
		starpu_data_assign_arbiter(A[i], arbiter);
	}

	for (loop = 0; loop < ITER; loop++)
	{
		for (i = 1; i < N-1; i++)
		{
			starpu_task_insert(&cl,
					STARPU_RW, B[i],
					STARPU_RW | STARPU_COMMUTE | STARPU_LOCALITY, A[i-1],
					STARPU_RW | STARPU_COMMUTE | STARPU_LOCALITY, A[i],
					STARPU_RW | STARPU_COMMUTE | STARPU_LOCALITY, A[i+1],
					STARPU_VALUE, &loop, sizeof(loop),
					STARPU_VALUE, &i, sizeof(i),
					0);
		}
	}

	starpu_task_wait_for_all();

	for (i = 0; i < N; i++)
	{
		starpu_data_unregister(A[i]);
		starpu_data_unregister(B[i]);
	}
	starpu_arbiter_destroy(arbiter);

	printf("worker where each domain piece was computed, over time\n");
	for (loop = 0; loop < ITER; loop++)
	{
		for (i = 1; i < N-1; i++)
		{
			printf("%02d ", task_worker[i][loop]);
		}
		printf("\n");
	}
	printf("\n");

	printf("domain piece that each worker has computed, over time\n");
	loop = 0;
	do
	{
		finished = 1;
		for (i = 0; i < starpu_worker_get_count(); i++)
		{
			if (loop < worker_ntask[i])
			{
				printf("%02d ", worker_task[i][loop]);
				finished = 0;
			}
			else
				printf("   ");
		}
		loop++;
		printf("\n");
	}
	while (!finished && loop < 100);

	starpu_shutdown();
	return EXIT_SUCCESS;
}
