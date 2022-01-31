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
#define PARTS 2
#define SIZE  24

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = PARTS
};

void sub_data_read_func(void *buffers[], void *arg)
{
}

void sub_data_func(void *buffers[], void *arg)
{
	int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
	int nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;

	for(i=0 ; i<nx ; i++)
		v[i] *= 2;
}

struct starpu_codelet sub_data_codelet =
{
	.cpu_funcs = {sub_data_func},
	.nbuffers = 1,
};

struct starpu_codelet sub_data_read_codelet =
{
	.cpu_funcs = {sub_data_read_func},
	.nbuffers = 1,
};

int is_bubble(struct starpu_task *t, void *arg)
{
	return 1;
}

void rec2_bubble_gen_dag(struct starpu_task *t, void *arg)
{
	unsigned i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_read_codelet,
					     STARPU_R, subdata[i],
					     STARPU_BUBBLE_PARENT, t,
					     STARPU_NAME, "B1_L3_task",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

starpu_data_handle_t sub_handles_l2[PARTS][PARTS];

void rec_bubble_gen_dag(struct starpu_task *t, void *arg)
{
	unsigned i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_read_codelet,
					     STARPU_R, subdata[i],
					     STARPU_BUBBLE_PARENT, t,
					     STARPU_BUBBLE_FUNC, &is_bubble,
					     STARPU_BUBBLE_GEN_DAG_FUNC, &rec2_bubble_gen_dag,
					     STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l2[i],
					     STARPU_NAME, "B1_L2",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
}

void bubble_gen_dag(struct starpu_task *t, void *arg)
{
	unsigned i;
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_read_codelet,
					     STARPU_R, subdata[i],
					     STARPU_BUBBLE_PARENT, t,
					     STARPU_NAME, "B2_L2_task",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
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

	starpu_data_handle_t main_handle;
	starpu_data_handle_t sub_handles_l1[PARTS];

	starpu_vector_data_register(&main_handle, STARPU_MAIN_RAM, (uintptr_t)v, SIZE, sizeof(v[0]));
	starpu_data_partition_plan(main_handle, &f, sub_handles_l1);

	for(i=0 ; i<PARTS ; i++)
	{
		starpu_data_partition_plan(sub_handles_l1[i], &f, sub_handles_l2[i]);
	}

	ret = starpu_task_insert(&sub_data_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "BEGIN",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&sub_data_read_codelet,
				 STARPU_R, main_handle,
				 STARPU_BUBBLE_FUNC, &is_bubble,
				 STARPU_BUBBLE_GEN_DAG_FUNC, &rec_bubble_gen_dag,
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 STARPU_NAME, "B1_L1",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&sub_data_read_codelet,
				 STARPU_R, main_handle,
				 STARPU_BUBBLE_FUNC, &is_bubble,
				 STARPU_BUBBLE_GEN_DAG_FUNC, &bubble_gen_dag,
				 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 STARPU_NAME, "B2_L1",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&sub_data_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "END",
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_task_wait_for_all();

	for(i=0 ; i<PARTS ; i++)
	{
		starpu_data_partition_clean(sub_handles_l1[i], PARTS, sub_handles_l2[i]);
	}
	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);
	starpu_shutdown();

//	for (i=0; i<SIZE; i++)
//	{
//		int x=(i+1);
//		int j;
//		for(j=0 ; j<STEPS ; j++) x*=2;
//		STARPU_ASSERT_MSG(v[i] == x, "Expected value %d != value %d", v[i], x);
//	}

	return 0;
}
