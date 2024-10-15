/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2019-2019  Gwenole Lucas
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

#define LESS_PARTS 3

struct b_args
{
	starpu_data_handle_t *sub1;
	starpu_data_handle_t *sub2;
	starpu_data_handle_t *sub3;
};

void recursive_task_gen_dag_ter(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Hello i am a recursive task\n");
	int i, ret;
	struct b_args *ba = (struct b_args *)arg;

	for(i=0 ; i<LESS_PARTS ; i++)
	{
		ret = starpu_task_insert(&sub_data_codelet,
					 STARPU_RW, ba->sub1[i],
					 STARPU_RECURSIVE_TASK_PARENT, t,
					 STARPU_NAME, "sub_data",
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		ret = starpu_task_insert(&sub_data_codelet,
					 STARPU_RW, ba->sub2[i],
					 STARPU_RECURSIVE_TASK_PARENT, t,
					 STARPU_NAME, "sub_data",
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		ret = starpu_task_insert(&sub_data_codelet,
					 STARPU_RW, ba->sub3[i],
					 STARPU_RECURSIVE_TASK_PARENT, t,
					 STARPU_NAME, "sub_data",
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet recursive_task_codelet_ter =
{
	.cpu_funcs = {recursive_task_func},
	.recursive_task_func = is_recursive_task,
	.recursive_task_gen_dag_func = recursive_task_gen_dag_ter,
	.nbuffers = 3,
	.model = &starpu_perfmodel_nop
};

void task_func_ter(void *buffers[], void *arg)
{
	int *v1 = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int *v2 = (int*)STARPU_VECTOR_GET_PTR(buffers[1]);
	int *v3 = (int*)STARPU_VECTOR_GET_PTR(buffers[2]);
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	size_t i;

	print_vector(v1, nx, "task");
	for(i=0 ; i<nx ; i++)
	{
		v1[i] += 10;
	}
}

struct starpu_codelet task_codelet_ter =
{
	.cpu_funcs = {task_func_ter},
	.nbuffers = 3
};

int main(int argv, char **argc)
{
	int ret, i;
	int v1[SIZE];
	int v2[SIZE];
	int v3[SIZE];

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
		v1[i] = i+1;
		v2[i] = 10*(i+1);
		v3[i] = 100*(i+1);
	}
	print_vector(v1, SIZE, "v init");

	starpu_data_handle_t handle1;
	starpu_data_handle_t handle2;
	starpu_data_handle_t handle3;
	starpu_data_handle_t sub_handles1_l1[PARTS];
	starpu_data_handle_t sub_handles2_l1[PARTS];
	starpu_data_handle_t sub_handles3_l1[PARTS];

	struct b_args ba;
	ba.sub1 = sub_handles1_l1;
	ba.sub2 = sub_handles2_l1;
	ba.sub3 = sub_handles3_l1;

	starpu_vector_data_register(&handle1, STARPU_MAIN_RAM, (uintptr_t)v1, SIZE, sizeof(v1[0]));
	starpu_vector_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)v2, SIZE, sizeof(v2[0]));
	starpu_vector_data_register(&handle3, STARPU_MAIN_RAM, (uintptr_t)v3, SIZE, sizeof(v3[0]));
	starpu_data_partition_plan(handle1, &f, sub_handles1_l1);
	starpu_data_partition_plan(handle2, &f, sub_handles2_l1);
	starpu_data_partition_plan(handle3, &f, sub_handles3_l1);

	ret = starpu_task_insert(&task_codelet_ter,
				 STARPU_RW, handle1,
				 STARPU_RW, handle2,
				 STARPU_RW, handle3,
				 STARPU_NAME, "T1", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&recursive_task_codelet_ter,
				 STARPU_RW, handle1,
				 STARPU_RW, handle2,
				 STARPU_RW, handle3,
				 STARPU_NAME, "B1",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, &ba,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&task_codelet_ter,
				 STARPU_RW, handle1,
				 STARPU_RW, handle2,
				 STARPU_RW, handle3,
				 STARPU_NAME, "T2", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&recursive_task_codelet_ter,
				 STARPU_RW, handle1,
				 STARPU_RW, handle2,
				 STARPU_RW, handle3,
				 STARPU_NAME, "B2",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, &ba,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&task_codelet_ter,
				 STARPU_RW, handle1,
				 STARPU_RW, handle2,
				 STARPU_RW, handle3,
				 STARPU_NAME, "T3", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_partition_clean(handle1, PARTS, sub_handles1_l1);
	starpu_data_partition_clean(handle2, PARTS, sub_handles2_l1);
	starpu_data_partition_clean(handle3, PARTS, sub_handles3_l1);
	starpu_data_unregister(handle1);
	starpu_data_unregister(handle2);
	starpu_data_unregister(handle3);
	starpu_shutdown();

	/* for (i=0; i<SIZE; i++) */
	/* { */
	/* 	int x=i+1; */
	/* 	check_task(x); check_recursive_task(x); check_task(x); check_recursive_task(x); check_task(x); */
	/* 	STARPU_ASSERT(v1[i] == x); */
	/* } */

	return 0;
}
