/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"

#define N 10
#define LOOPS 4

void null_cpu_func(void *buffers[], void *arg)
{
	(void)arg;
	(void)buffers;
}

void prod_cpu_func(void *buffers[], void *arg)
{
	int *data = (int *)STARPU_VECTOR_GET_PTR(buffers[0]);
	int n = STARPU_VECTOR_GET_NX(buffers[0]);
	int i;
	int factor;

	starpu_codelet_unpack_args(arg, &factor);

	FPRINTF(stderr, "Multiplying by %d\n", factor);
	for(i=0 ; i<n ; i++) data[i] *= factor;
}

static struct starpu_codelet cl_null =
{
	.cpu_funcs = {null_cpu_func},
	.cpu_funcs_name = {"null_cpu_func"},
	.model = &starpu_perfmodel_nop,
	.name = "null",
};

static struct starpu_codelet cl_prod =
{
	.cpu_funcs = {prod_cpu_func},
	.cpu_funcs_name = {"prod_cpu_func"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.flags = STARPU_CODELET_SIMGRID_EXECUTE,
	.model = &starpu_perfmodel_nop,
	.name = "prod",
};

int main(int argc, char **argv)
{
	int i, j, ret;
	int data[N];
	int data2[N];
	int factor[LOOPS];
	starpu_data_handle_t data_handle;

	ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	for(i=0 ; i<N ; i++) data[i] = 12;
	for(i=0 ; i<N ; i++) data2[i] = 12;
	starpu_vector_data_register(&data_handle, STARPU_MAIN_RAM, (uintptr_t) data, N, sizeof(int));

	struct starpu_task *motherTask = starpu_task_build(&cl_null, STARPU_NAME, "motherTask", 0);

	for (i = 0; i < LOOPS; i++)
	{
		factor[i] = i+1;
		for(j=0 ; j<N ; j++) data2[j] *= factor[i];
		ret = starpu_task_insert(&cl_prod,
					 STARPU_RW, data_handle,
					 STARPU_VALUE, &factor[i], sizeof(factor[i]),
					 STARPU_TASK_DEPS_ARRAY, 1, &motherTask,
					 0);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	ret = starpu_task_submit(motherTask);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	starpu_task_wait_for_all();
	starpu_data_unregister(data_handle);

	for(i=0 ; i<N ; i++)
	{
		FPRINTF(stderr, "data[%d] = %d ==? %d \n", i, data[i], data2[i]);
		STARPU_ASSERT_MSG(data[i] == data2[i], "Incorrect computation\n");
	}

	starpu_shutdown();
	return EXIT_SUCCESS;

enodev:
	fprintf(stderr, "WARNING: No one can execute this task\n");
	/* yes, we do not perform the computation but we did detect that no one
 	 * could perform the kernel, so this is not an error from StarPU */
	starpu_shutdown();
	return STARPU_TEST_SKIPPED;
}
