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
starpu_data_handle_t sub_handles_l2[PARTS][PARTS];

struct starpu_data_filter my_f =
{
        .filter_func = starpu_vector_filter_block,
        .nchildren = PARTS
};

void my_sub_data_func_ro(void *buffers[], void *arg)
{
        int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
        size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
	size_t i;
	print_vector(v, nx, "subsubtask_ro : ");
}


void my_sub_data_func(void *buffers[], void *arg)
{
        int *v = (int*)STARPU_VECTOR_GET_PTR(buffers[0]);
        size_t nx = STARPU_VECTOR_GET_NX(buffers[0]);
        size_t i;

        for(i=0 ; i<nx ; i++)
                v[i] *= 2;
	print_vector(v, nx, "subsubtask : ");
}

struct starpu_codelet my_sub_data_codelet =
{
        .cpu_funcs = {my_sub_data_func},
        .nbuffers = 1,
	.modes = {STARPU_RW}
};

struct starpu_codelet my_sub_data_codelet_ro =
{
        .cpu_funcs = {my_sub_data_func_ro},
        .nbuffers = 1,
	.modes = {STARPU_R}
};

int my_is_recursive_task(struct starpu_task *t, void *arg)
{
	return 1;
}

void recro2_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
        starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
        unsigned i;

        for(i=0 ; i<PARTS ; i++)
        {
                int ret = starpu_task_insert(&my_sub_data_codelet_ro,
                                             STARPU_R, subdata[i],
                                             STARPU_NAME, "Tro_L3",
                                             0);
                STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
        }
}

void recro_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
        starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
        unsigned i;

        for(i=0 ; i<PARTS ; i++)
        {
                int ret = starpu_task_insert(&my_sub_data_codelet_ro,
                                             STARPU_R, subdata[i],
                                             STARPU_RECURSIVE_TASK_FUNC, &my_is_recursive_task,
                                             STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &recro2_recursive_task_gen_dag,
                                             STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l2[i],
                                             STARPU_NAME, "Bro_L2",
                                             0);
                STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
        }
}


void rec2_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
        starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
        unsigned i;

        for(i=0 ; i<PARTS ; i++)
        {
                int ret = starpu_task_insert(&my_sub_data_codelet,
                                             STARPU_RW, subdata[i],
                                             STARPU_NAME, "T_L3",
                                             0);
                STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
        }
}

void rec_recursive_task_gen_dag(struct starpu_task *t, void *arg)
{
        starpu_data_handle_t *subdata = (starpu_data_handle_t *)arg;
        unsigned i;

        for(i=0 ; i<PARTS ; i++)
        {
                int ret = starpu_task_insert(&my_sub_data_codelet,
                                             STARPU_RW, subdata[i],
                                             STARPU_RECURSIVE_TASK_FUNC, &my_is_recursive_task,
                                             STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &rec2_recursive_task_gen_dag,
                                             STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l2[i],
                                             STARPU_NAME, "B_L2",
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
	print_vector(v, SIZE, "v init");

        starpu_data_handle_t main_handle;
        starpu_data_handle_t sub_handles_l1[PARTS];

        starpu_vector_data_register(&main_handle, STARPU_MAIN_RAM, (uintptr_t)v, SIZE, sizeof(v[0]));
        starpu_data_partition_plan(main_handle, &my_f, sub_handles_l1);

        for(i=0 ; i<PARTS ; i++)
        {
                starpu_data_partition_plan(sub_handles_l1[i], &my_f, sub_handles_l2[i]);
        }

	ret = starpu_task_insert(&my_sub_data_codelet_ro,
				 STARPU_R, main_handle,
				 STARPU_NAME, "Bro_L1",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &recro_recursive_task_gen_dag,
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 STARPU_RECURSIVE_TASK_FUNC, &my_is_recursive_task,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	ret = starpu_task_insert(&my_sub_data_codelet,
				 STARPU_RW, main_handle,
				 STARPU_NAME, "B_L1",
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC, &rec_recursive_task_gen_dag,
				 STARPU_RECURSIVE_TASK_GEN_DAG_FUNC_ARG, sub_handles_l1,
				 STARPU_RECURSIVE_TASK_FUNC, &my_is_recursive_task,
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

	for (i=0; i<SIZE; i++)
	{
		int x=i+1;
		check_recursive_task(x);
		STARPU_ASSERT(v[i] == x);
	}

	return 0;
}
