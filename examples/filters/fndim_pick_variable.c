/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010       Mehdi Juhoor
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

#define NX    21
#define PARTS 3
#define POS   5

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	int *factor = (int *) cl_arg;

	/* local copy of the variable pointer */
	int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);

	*val *= *factor;
}

int main(void)
{
	int i;
	int *arr1d;
	starpu_data_handle_t handle;
	int factor = 10;
	int ret;

	struct starpu_codelet cl =
	{
		.cpu_funcs = {cpu_func},
		.cpu_funcs_name = {"cpu_func"},
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "arr1d_pick_variable_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
	exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&arr1d, NX*sizeof(int));
	FPRINTF(stderr,"IN 1-dim Array: \n");
	for(i=0 ; i<NX ; i++)
	{
		arr1d[i] = i;
		FPRINTF(stderr, "%5d ", arr1d[i]);
	}
	FPRINTF(stderr,"\n");

	unsigned nn[1] = {NX};
	unsigned ldn[1] = {1};

	/* Declare data to StarPU */
	starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)arr1d, ldn, nn, 1, sizeof(int));

	/* Partition the 1-dim array in PARTS sub-variables */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_ndim_filter_pick_variable,
		.filter_arg_ptr = (void*)(uintptr_t) POS,
		.nchildren = PARTS,
		/* the children use a variable interface*/
		.get_child_ops = starpu_ndim_filter_pick_variable_child_ops
	};
	starpu_data_partition(handle, &f);

	/* Submit a task on each sub-variable */
	for (i=0; i<starpu_data_get_nb_children(handle); i++)
	{
		starpu_data_handle_t variable_handle = starpu_data_get_sub_data(handle, 1, i);
		FPRINTF(stderr, "Sub Variable %d: \n", i);
		int *variable = (int *)starpu_variable_get_local_ptr(variable_handle);
		FPRINTF(stderr, "%5d ", *variable);
		FPRINTF(stderr,"\n");

		struct starpu_task *task = starpu_task_create();
		FPRINTF(stderr,"Dealing with sub-variable %d\n", i);
		task->handles[0] = variable_handle;
		task->cl = &cl;
		task->synchronous = 1;
		task->cl_arg = &factor;
		task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		FPRINTF(stderr,"OUT Variable %d: \n", i);
		FPRINTF(stderr, "%5d ", *variable);
		FPRINTF(stderr,"\n");
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unregister(handle);

	FPRINTF(stderr,"OUT 1-dim Array: \n");
	for(i=0 ; i<NX ; i++) FPRINTF(stderr, "%5d ", arr1d[i]);
	FPRINTF(stderr,"\n");

	starpu_free_noflag(arr1d, NX*sizeof(int));

	starpu_shutdown();

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
