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

#define ITER 3

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

	ret = starpu_task_insert(&task_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "T0",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&recursive_task_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "B0",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	for (i=1; i<ITER+1; i++)
	{
		char *recursive_task_name, *task_name;
		asprintf(&recursive_task_name, "B_%d", i);
		asprintf(&task_name, "T_%d", i);
		ret = starpu_task_insert(&recursive_taskRO_codelet,
					 STARPU_R, main_handle,
					 STARPU_NAME, recursive_task_name,
					 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		ret = starpu_task_insert(&taskRO_codelet,
					 STARPU_R, main_handle,
					 STARPU_NAME, task_name, 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();

	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);
	starpu_shutdown();

	for (i=0; i<SIZE; i++)
	{
		int x=i+1;
		check_task(x);check_recursive_task(x);
		STARPU_ASSERT(v[i] == x);
	}


	return 0;
}
