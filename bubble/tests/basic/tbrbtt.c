/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019       Gwenole Lucas
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
#include "basic.h"

#define LENGTH 16
#define NPARTS 2

struct handle_partition
{
	starpu_data_handle_t handle;
	starpu_data_handle_t *sub;
	starpu_data_handle_t *sub0;
	starpu_data_handle_t *sub1;
};

struct starpu_data_filter filter =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = NPARTS
};

void task_2arg_func(void *buffers[], void *arg)
{
	int *v1 = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int *v2 = (int*)STARPU_VECTOR_GET_PTR(buffers[1]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	print_vector(v1, nx, "task");
	for(i=0 ; i<nx ; i++)
	{
		v1[i] += v2[i];
	}
}

struct starpu_codelet task_2arg_codelet =
{
	.cpu_funcs = {task_2arg_func},
	.nbuffers = 2
};

void bubble_2arg_gen_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Bubble level 2\n");
	struct handle_partition *handles = (struct handle_partition*)arg;

	int ret;
	ret = starpu_task_insert(&task_2arg_codelet,
				 STARPU_R, handles->sub0[1],
				 STARPU_RW, handles->sub1[0],
				 STARPU_NAME, "Task",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&task_2arg_codelet,
				 STARPU_R, handles->sub0[1],
				 STARPU_RW, handles->sub1[1],
				 STARPU_NAME, "Task",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

struct starpu_codelet bubble_2arg_codelet =
{
	.cpu_funcs = {bubble_func},
	.bubble_func = is_bubble,
	.bubble_gen_dag_func = bubble_2arg_gen_dag,
	.nbuffers = 2
};

void bubble_1arg_gen_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Bubble level 1\n");
	struct handle_partition *handles = (struct handle_partition*)arg;

	int ret = starpu_task_insert(&bubble_2arg_codelet,
				     STARPU_R, handles->sub[0],
				     STARPU_RW, handles->sub[1],
				     STARPU_NAME, "BubbleLvl2",
				     STARPU_BUBBLE_GEN_DAG_FUNC_ARG, handles,
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

struct starpu_codelet bubble_1arg_codelet =
{
	.cpu_funcs = {bubble_func},
	.bubble_func = is_bubble,
	.bubble_gen_dag_func = bubble_1arg_gen_dag,
	.nbuffers = 1
};

int main(int argv, char **argc)
{
	int ret, i;
	int v[LENGTH];

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_shutdown();
		return 77;
	}

	for (i=0; i<LENGTH; i++)
	{
		v[i] = i+1;
	}
	print_vector(v, LENGTH, "v init");


	starpu_data_handle_t A, B;
	starpu_data_handle_t subA[NPARTS];
	starpu_data_handle_t subA0[NPARTS];
	starpu_data_handle_t subA1[NPARTS];

	/* Level 0 */
	starpu_vector_data_register(&A, STARPU_MAIN_RAM, (uintptr_t)v, LENGTH, sizeof(v[0]));
	starpu_vector_data_register(&B, STARPU_MAIN_RAM, (uintptr_t)v, LENGTH, sizeof(v[0]));

	/* Level 1 */
	starpu_data_partition_plan(A, &filter, subA);

	/* Level 2 */
	starpu_data_partition_plan(subA[0], &filter, subA0);
	starpu_data_partition_plan(subA[1], &filter, subA1);

	struct handle_partition bubble_arg =
	{
		.handle = A,
		.sub = subA,
		.sub0 = subA0,
		.sub1 = subA1
	};

	ret = starpu_task_insert(&task_2arg_codelet,
				 STARPU_R, A,
				 STARPU_RW, B,
				 STARPU_NAME, "Task",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&bubble_1arg_codelet,
				 STARPU_RW, A,
				 STARPU_NAME, "BubbleLvl1",
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, &bubble_arg,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_partition_clean(subA[0], NPARTS, subA0);
	starpu_data_partition_clean(subA[1], NPARTS, subA1);
	starpu_data_partition_clean(A, NPARTS, subA);
	starpu_data_unregister(A);
	starpu_data_unregister(B);
	starpu_shutdown();

	return 0;
}
