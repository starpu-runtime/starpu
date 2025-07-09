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

int is_recursive_task_strong(struct starpu_task *t, void *arg, void **buffers)
{
	(void)arg;
	(void)t;
	int *v = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
	size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	size_t i;
	print_vector(v, SIZE, "v decide");
	for (i=0; i < nx; i++)
		if (v[i] == (int)i+1)
			return 1;
	print_vector(v, SIZE, "v decide");
	return 0;
}

struct starpu_codelet recursive_task_strongdeps_codelet =
{
	.cpu_funcs = {task_func},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {task_cuda_func},
#endif
	.recursive_task_func = is_recursive_task_strong,
	.recursive_task_gen_dag_func = recursive_task_gen_dag,
	.nbuffers = 1,
	.model = &starpu_perfmodel_nop
};

int main(int argv, char **argc)
{
	int ret, i;
	int v[SIZE];

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

	ret = starpu_task_insert(&recursive_task_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "B1",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&recursive_task_strongdeps_codelet,
				 STARPU_RW | STARPU_RECURSIVE_TASK_STRONG, main_handle,
				 STARPU_NAME, "B2",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();
	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);
	starpu_shutdown();

	for (i=0; i<SIZE; i++)
	{
		int x=i+1;
		check_recursive_task(x); check_task(x);
		STARPU_ASSERT(v[i] == x);
	}


	return 0;
}
