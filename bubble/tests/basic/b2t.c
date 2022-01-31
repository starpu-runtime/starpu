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

#define check_binary_task(x,y) x+=y

struct starpu_codelet sub_data_chain_codelet =
{
	.cpu_funcs = {sub_data_func},
	.nbuffers = 2,
	.name = "sub_data_chain_cl"
};

void bubble_chain_gen_dag(struct starpu_task *t, void *arg)
{
	FPRINTF(stderr, "Hello i am a bubble\n");
	int i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_chain_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_RW, subdata[0], /* Just to chain the tasks submitted by the bubble */
					     STARPU_NAME, "T'(subA)",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet bubble_chain_codelet =
{
	.cpu_funcs = {bubble_func},
	.bubble_func = is_bubble,
	.bubble_gen_dag_func = bubble_chain_gen_dag,
	.nbuffers = 1
};

void binary_task_func(void *buffers[], void *arg)
{
	int *vA = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int *vB = (int*)STARPU_VECTOR_GET_PTR(buffers[1]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	print_vector(vA, nx, "task vA");
	for(i=0 ; i<nx ; i++)
	{
		vB[i] += vA[i];
	}
}

struct starpu_codelet binary_task_codelet =
{
	.cpu_funcs = {binary_task_func},
	.nbuffers = 2
};

int main(int argv, char **argc)
{
	int ret, i;
	int vA[SIZE];
	int vB[SIZE];
	int vC[SIZE];

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
		vA[i] = 2*(i+1);
		vB[i] = 2*i+1;
		vC[i] = 2*i+1;
	}
	print_vector(vA, SIZE, "vA init");
	print_vector(vB, SIZE, "vB init");
	print_vector(vC, SIZE, "vC init");

	starpu_data_handle_t A, B, C;
	starpu_data_handle_t subA[PARTS];

	starpu_vector_data_register(&A, STARPU_MAIN_RAM, (uintptr_t)vA, SIZE, sizeof(vA[0]));
	starpu_vector_data_register(&B, STARPU_MAIN_RAM, (uintptr_t)vB, SIZE, sizeof(vB[0]));
	starpu_vector_data_register(&C, STARPU_MAIN_RAM, (uintptr_t)vC, SIZE, sizeof(vC[0]));

	starpu_data_partition_plan(A, &f, subA);

	ret = starpu_task_insert(&bubble_chain_codelet,
				 STARPU_RW, A,
				 STARPU_NAME, "B(A)",
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, subA,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&binary_task_codelet,
				 STARPU_R,  A,
				 STARPU_RW, B,
				 STARPU_NAME, "T(A,B)", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&binary_task_codelet,
				 STARPU_R,  A,
				 STARPU_RW, C,
				 STARPU_NAME, "T(A,C)", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_partition_clean(A, PARTS, subA);
	starpu_data_unregister(A);
	starpu_data_unregister(B);
	starpu_data_unregister(C);
	starpu_shutdown();

	print_vector(vA, SIZE, "vA final");
	print_vector(vB, SIZE, "vB final");
	print_vector(vC, SIZE, "vC final");

	int check[SIZE];
	for (i=0; i<SIZE; i++)
	{
		int a = 2*(i+1); check[i] = 2*i+1;
		check_bubble(a); check_binary_task(check[i], a);
		STARPU_ASSERT(vB[i] == check[i]);
	}
	FPRINTF(stderr, "vB is correct\n");
	for (i=0; i<SIZE; i++)
	{
		STARPU_ASSERT(vC[i] == check[i]);
	}
	FPRINTF(stderr, "vC is correct\n");

	return 0;
}
