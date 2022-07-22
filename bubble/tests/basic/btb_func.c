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

struct starpu_codelet my_codelet;

void my_task_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	print_vector(v, nx, "task");
	for(i=0 ; i<nx ; i++)
	{
		v[i] += 10;
	}
}

int is_a_bubble(struct starpu_task *t, void *_arg)
{
	int *arg = (int *)_arg;
	if (arg) return *arg; else return 0;
}

void bubble_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Hello i am a bubble\n");
	int i=0;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&my_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_NAME, "sub_data",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet my_codelet =
{
	.cpu_funcs = {my_task_func},
	.bubble_func = is_a_bubble,
	.bubble_gen_dag_func = bubble_dag,
	.nbuffers = 1
};

int main(int argv, char **argc)
{
	int ret, i;
	int v[SIZE];
	int is_bubble_p = 1;
	int is_task = 0;

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
		v[i] = i+1;
	}
	print_vector(v, SIZE, "v init");

	starpu_data_handle_t main_handle;
	starpu_data_handle_t sub_handles_l1[PARTS];

	starpu_vector_data_register(&main_handle, STARPU_MAIN_RAM, (uintptr_t)v, SIZE, sizeof(v[0]));
	starpu_data_partition_plan(main_handle, &f, sub_handles_l1);

	ret = starpu_task_insert(&my_codelet,
				 STARPU_RW, main_handle,
				 STARPU_BUBBLE_FUNC_ARG, (void *)&is_bubble_p,
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 STARPU_NAME, "B1", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&my_codelet,
				 STARPU_RW, main_handle,
				 STARPU_BUBBLE_FUNC_ARG, (void *)&is_task,
				 STARPU_NAME, "T1", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&my_codelet,
				 STARPU_RW, main_handle,
				 STARPU_BUBBLE_FUNC_ARG, (void *)&is_bubble_p,
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 STARPU_NAME, "B2", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);
	starpu_shutdown();

	for (i=0; i<SIZE; i++)
	{
		int x=i+1;
		check_task(x); check_task(x); check_task(x);
		STARPU_ASSERT(v[i] == x);
	}

	return 0;
}
