/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2023-2023  Gwenole Lucas
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

#define PARTS 2
#define SIZE 64
#include "basic.h"

starpu_data_handle_t sub_handles_l3[PARTS];

void rec3_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	FPRINTF(stderr, "Hello i am a recursive task\n");

	int ret = starpu_task_insert(&sub_data_codelet,
				     STARPU_RW, subdata[0],
				     STARPU_RECURSIVE_TASK_PARENT, t,
				     STARPU_NAME, "T3-RW",
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

void rec2_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	FPRINTF(stderr, "Hello i am a recursive task\n");

	int ret = starpu_task_insert(&sub_data_codelet,
				     STARPU_RW, subdata[0],
				     STARPU_RECURSIVE_TASK_PARENT, t,
				     STARPU_NAME, "T2-RW",
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_insert(&recursive_task_codelet,
			   STARPU_RW, subdata[0],
			   STARPU_RECURSIVE_TASK_PARENT, t,
			   STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &rec3_recursive_task_gen_dag,
			   STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l3,
			   STARPU_NAME, "B2-RW",
			   0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

starpu_data_handle_t sub_handles_l2[PARTS];

void rec_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	FPRINTF(stderr, "Hello i am a double recursive task\n");

	int ret = starpu_task_insert(&task_codelet,
				     STARPU_RW, subdata[0],
				     STARPU_RECURSIVE_TASK_PARENT, t,
				     STARPU_NAME, "T1-RW",
				     0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_insert(&recursive_task_codelet,
			   STARPU_RW, subdata[0],
			   STARPU_RECURSIVE_TASK_PARENT, t,
			   STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &rec2_recursive_task_gen_dag,
			   STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l2,
			   STARPU_NAME, "B1-RW",
			   0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

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
	starpu_data_partition_plan(sub_handles_l1[0], &f, sub_handles_l2);
	starpu_data_partition_plan(sub_handles_l2[0], &f, sub_handles_l3);

	int steps = 1;//1000;
	for(i=0 ; i<steps ; i++)
	{
		ret = starpu_task_insert(&recursive_task_codelet,
					 STARPU_RW, main_handle,
					 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &rec_recursive_task_gen_dag,
					 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
					 STARPU_NAME, "B0-RW", 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		ret = starpu_task_insert(&task_codelet,
					 STARPU_RW, main_handle,
					 STARPU_NAME, "T0-RW", 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		ret = starpu_task_insert(&recursive_task_codelet,
					 STARPU_RW, main_handle,
					 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &rec_recursive_task_gen_dag,
					 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
					 STARPU_NAME, "B0-RW", 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

		ret = starpu_task_insert(&taskRO_codelet,
					 STARPU_R, main_handle,
					 STARPU_NAME, "T0-RO", 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();

	starpu_data_partition_clean(sub_handles_l2[0], PARTS, sub_handles_l3);
	starpu_data_partition_clean(sub_handles_l1[0], PARTS, sub_handles_l2);
	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);
	starpu_shutdown();

	return 0;
}
