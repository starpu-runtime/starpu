/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void set(void *buffers[], void *cl_arg)
{
	unsigned i;
	(void)cl_arg;

	unsigned nx = STARPU_VECTOR_GET_NX(buffers[0]);
	int *val = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);

	for(i=0; i<nx ; i++)
		val[i] = i+100;
}

struct starpu_codelet cl_set =
{
	.cpu_funcs = {set},
	.cpu_funcs_name = {"set"},
	.nbuffers = 1,
	.modes = {STARPU_W},
	.name = "set",
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
};

#define NX    12

int main(void)
{
	unsigned n=1;
	int vector[NX];
	int ret, i;

	starpu_data_handle_t handle;
	starpu_data_handle_t sub_handles_1[2];
	starpu_data_handle_t sub_handles_2[3];

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

	struct starpu_data_filter filter1 =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = 2
	};
	starpu_data_partition_plan(handle, &filter1, sub_handles_1);

	struct starpu_data_filter filter2 =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = 3
	};
	starpu_data_partition_plan(handle, &filter2, sub_handles_2);

	for (i = 0; i < 2; i++)
	{
		ret = starpu_task_insert(&cl_set,
					 STARPU_W, sub_handles_1[i],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	for (i = 0; i < 3; i++)
	{
		ret = starpu_task_insert(&cl_set,
					 STARPU_W, sub_handles_2[i],
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_data_partition_clean(handle, 2, sub_handles_1);
	starpu_data_partition_clean(handle, 3, sub_handles_2);
	starpu_data_unregister(handle);
	starpu_shutdown();

	return ret;

enodev:
	starpu_shutdown();
	return 77;
}
