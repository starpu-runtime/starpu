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

#define check_binary_task(x,y) x+=y

int main(int argv, char **argc)
{
	int ret, i;
	int vA[SIZE];

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
		vA[i] = i+1;
	}
	print_vector(vA, SIZE, "vA init");

	starpu_data_handle_t A;
	starpu_data_handle_t subA[PARTS];

	starpu_vector_data_register(&A, STARPU_MAIN_RAM, (uintptr_t)vA, SIZE, sizeof(vA[0]));

	starpu_data_partition_plan(A, &f, subA);

	/* starpu_pause(); */

	ret = starpu_task_insert(&recursive_taskRO_codelet,
				 STARPU_R, A,
				 STARPU_NAME, "B",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, subA,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&recursive_taskRO_codelet,
				 STARPU_R, A,
				 STARPU_NAME, "B'",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, subA,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&task_codelet,
				 STARPU_RW, A,
				 STARPU_NAME, "T",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	/* starpu_resume(); */

	starpu_task_wait_for_all();

	starpu_data_partition_clean(A, PARTS, subA);
	starpu_data_unregister(A);
	starpu_shutdown();

	print_vector(vA, SIZE, "vA final");

	return 0;
}
