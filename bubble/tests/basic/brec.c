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
#define PARTS 1
#define SIZE  25

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

struct starpu_data_filter f =
{
	.filter_func = starpu_vector_filter_block,
	.nchildren = PARTS
};

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

int rec_is_bubble(struct starpu_task *t, void *arg)
{
	int v = *(int *)arg;
	fprintf(stderr, "'%s' is a %s\n", starpu_task_get_name(t), (v == 0)?"task":"bubble");
	free(arg);
	return v;
}

void rec2_bubble_gen_dag(struct starpu_task *t, void *arg)
{
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
	int i;

	for(i=0 ; i<PARTS ; i++)
	{
		int ret = starpu_task_insert(&sub_data_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_NAME, "T_L3",
					     0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	}
}

void free_memory(void *arg)
{
	free(arg);
}

starpu_data_handle_t sub_handles_l2[PARTS][PARTS];

void rec_bubble_gen_dag(struct starpu_task *t, void *arg)
{
	starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
	unsigned i;

	for(i=0 ; i<PARTS ; i++)
	{
		int *is_bubble = malloc(sizeof(int));
		*is_bubble = random() & 1;
		char *name;
		asprintf(&name, "%s %s", starpu_task_get_name(t), (*is_bubble == 0) ? "T_L2" : "B_L2");

		int ret = starpu_task_insert(&sub_data_codelet,
					     STARPU_RW, subdata[i],
					     STARPU_BUBBLE_PARENT, t,
					     STARPU_BUBBLE_FUNC, &rec_is_bubble,
					     STARPU_BUBBLE_FUNC_ARG, is_bubble,
					     STARPU_BUBBLE_GEN_DAG_FUNC, &rec2_bubble_gen_dag,
					     STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l2[i],
					     STARPU_CALLBACK_WITH_ARG_NFREE, &free_memory, name,
					     STARPU_NAME, name,
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

#define STEPS 10

	for(i=0 ; i<STEPS ; i++)
	{
		int *is_bubble = malloc(sizeof(int));
		*is_bubble = random() & 1;

		char *name;
		asprintf(&name, "%s %d", (*is_bubble == 0) ? "T_L1" : "B_L1", i);

		ret = starpu_task_insert(&sub_data_codelet,
					 STARPU_RW, main_handle,
					 STARPU_BUBBLE_FUNC, &rec_is_bubble,
					 STARPU_BUBBLE_FUNC_ARG, is_bubble,
					 STARPU_BUBBLE_GEN_DAG_FUNC, &rec_bubble_gen_dag,
					 STARPU_BUBBLE_GEN_DAG_FUNC_ARG, sub_handles_l1,
					 STARPU_NAME, name,
					 STARPU_CALLBACK_WITH_ARG_NFREE, &free_memory, name,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_task_wait_for_all();

	for(i=0 ; i<PARTS ; i++)
	{
		starpu_data_partition_clean(sub_handles_l1[i], PARTS, sub_handles_l2[i]);
	}
	starpu_data_partition_clean(main_handle, PARTS, sub_handles_l1);
	starpu_data_unregister(main_handle);

	for (i=0; i<SIZE; i++)
	{
		int x=(i+1);
		int j;
		for(j=0 ; j<STEPS ; j++) x*=2;
		STARPU_ASSERT_MSG(v[i] == x, "Expected value %d != value %d", v[i], x);
	}

	starpu_shutdown();

	return 0;
}
