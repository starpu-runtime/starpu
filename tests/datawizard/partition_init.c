/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2019-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdio.h>
#include <starpu.h>
#include "../helper.h"

void my_func(void *buffers[], void *cl_arg)
{
	unsigned nb = STARPU_VECTOR_GET_NX(buffers[0]);
        int *v = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

	unsigned i;
	for(i=0 ; i<nb ; i++)
	{
		v[i] = i+42;
		FPRINTF(stderr, "setting v[%u] to %d\n",i, v[i]);
	}
}

struct starpu_codelet my_codelet =
{
	.cpu_funcs = {my_func},
	.nbuffers = 1,
	.modes = {STARPU_W}
};

void display_func(void *buffers[], void *cl_arg)
{
	unsigned nb = STARPU_VECTOR_GET_NX(buffers[0]);
        int *v = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

	unsigned i;
	for(i=0 ; i<nb ; i++) FPRINTF(stderr, "v[%u] = %d\n", i, v[i]);
}

struct starpu_codelet display_codelet =
{
	.cpu_funcs = {display_func},
	.nbuffers = 1,
	.modes = {STARPU_R}
};

int main(int argc, char **argv)
{
	int ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (starpu_cpu_worker_get_count() == 0)
	{
		FPRINTF(stderr, "The test needs at least 1 CPU worker\n");
		starpu_shutdown();
		return STARPU_TEST_SKIPPED;
	}

	struct starpu_data_filter f =
	{
	 	.filter_func = starpu_vector_filter_block,
		.nchildren = starpu_cpu_worker_get_count()
	};

	starpu_data_handle_t array_handle;
	starpu_vector_data_register(&array_handle, -1, (uintptr_t)NULL, f.nchildren*2, sizeof(int));
	starpu_data_partition(array_handle, &f);

	int i;
	for(i=0 ; i<starpu_data_get_nb_children(array_handle) ; i++)
	{
		starpu_data_handle_t sub_handle = starpu_data_get_sub_data(array_handle, 1, i);
		ret = starpu_task_insert(&my_codelet,
					 STARPU_W, sub_handle,
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}
	starpu_data_unpartition(array_handle, STARPU_MAIN_RAM);

	ret = starpu_task_insert(&display_codelet,
				 STARPU_R, array_handle,
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

	starpu_data_unregister(array_handle);
	starpu_shutdown();

	return 0;
}
