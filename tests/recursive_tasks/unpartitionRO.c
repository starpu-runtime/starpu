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

#define N 3

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
		v[i] = i+1;
	print_vector(v, SIZE, "vA init");

	starpu_data_handle_t main_handle;
	starpu_data_handle_t sub_handles_l1[PARTS];
	starpu_data_handle_t sub_handles_l2[PARTS][PARTS];

	starpu_vector_data_register(&main_handle, STARPU_MAIN_RAM, (uintptr_t)v, SIZE, sizeof(v[0]));
	starpu_data_partition_plan(main_handle, &f, sub_handles_l1);
	for (i=0; i<PARTS; i++)
		starpu_data_partition_plan(sub_handles_l1[i], &f, sub_handles_l2[i]);

	/* We want to prevent subtask execution until the recursive_task has finished */
//	starpu_pause();

	for (i=0; i<N-1; i++)
	{
		ret = starpu_task_insert(&recursive_task_codelet,
					 STARPU_RW, main_handle,
					 STARPU_NAME, "B",
					 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	ret = starpu_task_insert(&recursive_task_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "B",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
//				 STARPU_CALLBACK, starpu_resume,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	for (i=0; i<N; i++)
	{
		ret = starpu_task_insert(&taskRO_codelet,
					 STARPU_R, main_handle,
					 STARPU_NAME, "T_RO",
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	ret = starpu_task_insert(&recursive_task_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "B2",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	for (i=0; i<PARTS; i++)
		starpu_data_partition_clean(sub_handles_l1[i], PARTS, sub_handles_l2[i]);
	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);
	starpu_shutdown();

	print_vector(v, SIZE, "vA final");

	return 0;
}
