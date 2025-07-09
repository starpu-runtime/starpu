/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

struct starpu_codelet sub_data_chain_codelet =
{
	.cpu_funcs = {sub_data_func},
	.nbuffers = 1,
	.name = "sub_data_chain_cl"
};

void recursive_task_chain_gen_dag(struct starpu_task *t, void *arg, void **b)
{
	FPRINTF(stderr, "Hello i am a recursive task\n");
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
	int i;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_chain_codelet,
					     STARPU_R, subdata[i],
					     STARPU_NAME, "T'(subA)",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

struct starpu_codelet recursive_task_chain_codelet =
{
	.cpu_funcs = {recursive_task_func},
	.recursive_task_func = is_recursive_task_always,
	.recursive_task_gen_dag_func = recursive_task_chain_gen_dag,
	.nbuffers = 1
};

void btask_func(void *buffers[], void *arg)
{
	int *vA = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);

	print_vector(vA, nx, "task vA");
}

struct starpu_codelet btask_codelet =
{
	.cpu_funcs = {btask_func},
	.nbuffers = 1
};

int main(int argv, char **argc)
{
	int ret;
	int vA[SIZE] = { 0 };

	ret = starpu_init(NULL);
	if (ret == -ENODEV) return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "We need at least 1 CPU worker.\n");
		starpu_shutdown();
		return 77;
	}

	starpu_data_handle_t A;
	starpu_data_handle_t subA[PARTS];

	starpu_vector_data_register(&A, STARPU_MAIN_RAM, (uintptr_t)vA, SIZE, sizeof(vA[0]));
	starpu_data_partition_plan(A, &f, subA);

	ret = starpu_task_insert(&sub_data_codelet,
				 STARPU_RW,  A,
				 STARPU_NAME, "T0", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&recursive_task_chain_codelet,
				 STARPU_R, A,
				 STARPU_NAME, "B",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, subA,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&btask_codelet,
				 STARPU_R,  A,
				 STARPU_NAME, "T1", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&btask_codelet,
				 STARPU_R,  A,
				 STARPU_NAME, "T2", 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	starpu_data_partition_clean(A, PARTS, subA);
	starpu_data_unregister(A);
	starpu_shutdown();

	return 0;
}
