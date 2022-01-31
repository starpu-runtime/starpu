/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#define PARTS 2
#define SIZE  16

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = PARTS
};

void scam_func(void *buffers[], void *arg)
{
	assert(0);
}

void real_func(void *buffers[], void *arg)
{
	int *A = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;
	for (i=0; i<nx; i++)
	{
		FPRINTF(stderr, "%d ", A[i]);
	}
	FPRINTF(stderr, "\n");
}

struct starpu_codelet scam_codelet =
{
	.cpu_funcs = {scam_func},
	.nbuffers = 1
};

struct starpu_codelet real_codelet =
{
	.cpu_funcs = {real_func},
	.nbuffers = 1
};

int always_bubble(struct starpu_task *t, void *arg)
{
	return 1;
}

void bubble_gen_dag_func(struct starpu_task *t, void *arg)
{
	int i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&real_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_TASK_SYNCHRONOUS, 1,
					     STARPU_NAME, "task",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

int main(int argv, char *argc[])
{
	int ret, i;
	int A[SIZE];
	starpu_data_handle_t handle;
	starpu_data_handle_t subhandles[PARTS];

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_shutdown();
		return 77;
	}

	for (i=0; i<SIZE; i++)
	{
		A[i] = i;
	}

	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&A, SIZE, sizeof(A[0]));
	starpu_data_partition_plan(handle, &f, subhandles);

	/* insert bubble on handle */
	ret = starpu_task_insert(&scam_codelet,
				 STARPU_RW, handle,
				 STARPU_BUBBLE_FUNC, always_bubble,
				 STARPU_BUBBLE_GEN_DAG_FUNC, bubble_gen_dag_func,
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, subhandles,
				 STARPU_TASK_SYNCHRONOUS, 1,
				 STARPU_NAME, "bubble",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_partition_clean(handle, PARTS, subhandles);
	starpu_data_unregister(handle);
	starpu_shutdown();

	return 0;
}
