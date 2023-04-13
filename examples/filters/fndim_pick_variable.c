/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define NX    6
#define NY    5
#define NZ    4
#define NT    3
#define NG    2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	int *factor = (int *) cl_arg;

	/* local copy of the variable pointer */
	int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);

	*val *= *factor;
}

extern void generate_5dim_data(int *arr5d, int nx, int ny, int nz, int nt, int ng, unsigned ldy, unsigned ldz, unsigned ldt, unsigned ldg);
extern void print_5dim_data(starpu_data_handle_t ndim_handle);

int main(void)
{
	int *arr5d;
	int i, j, k, l, m;
	int ret;
	int factor = 2;
	uint32_t pos[5] = {1,2,1,2,1};

	starpu_data_handle_t handle;
	starpu_data_handle_t var_handle[1];

	struct starpu_codelet cl =
	{
		.cpu_funcs = {cpu_func},
		.cpu_funcs_name = {"cpu_func"},
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "arr5d_pick_variable_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&arr5d, NX*NY*NZ*NT*NG*sizeof(int));
	assert(arr5d);
	generate_5dim_data(arr5d, NX, NY, NZ, NT, NG, NX, NX*NY, NX*NY*NZ, NX*NY*NZ*NT);

	unsigned nn[5] = {NX, NY, NZ, NT, NG};
	unsigned ldn[5] = {1, NX, NX*NY, NX*NY*NZ, NX*NY*NZ*NT};

	/* Declare data to StarPU */
	starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)arr5d, ldn, nn, 5, sizeof(int));
	FPRINTF(stderr, "IN 5-dim Array: \n");
	print_5dim_data(handle);

	/* Pick a variable in the 5-dim array */
	struct starpu_data_filter f_var =
	{
		.filter_func = starpu_ndim_filter_pick_variable,
		.filter_arg_ptr = (void*)pos,
		.nchildren = 1,
		/* the children use a variable interface*/
		.get_child_ops = starpu_ndim_filter_pick_variable_child_ops
	};

	starpu_data_partition_plan(handle, &f_var, var_handle);

	FPRINTF(stderr, "Sub Variable:\n");
	int *variable = (int *)starpu_variable_get_local_ptr(var_handle[0]);
	starpu_data_acquire(var_handle[0], STARPU_R);
	FPRINTF(stderr, "%5d ", *variable);
	starpu_data_release(var_handle[0]);
	FPRINTF(stderr,"\n");

	/* Submit the task */
	struct starpu_task *task = starpu_task_create();

	FPRINTF(stderr,"Dealing with sub-variable\n");
	task->handles[0] = var_handle[0];
	task->cl = &cl;
	task->synchronous = 1;
	task->cl_arg = &factor;
	task->cl_arg_size = sizeof(factor);

	ret = starpu_task_submit(task);
	if (ret == -ENODEV) goto enodev;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

	/* Print result variable */
	FPRINTF(stderr,"OUT Variable:\n");
	starpu_data_acquire(var_handle[0], STARPU_R);
	FPRINTF(stderr, "%5d ", *variable);
	starpu_data_release(var_handle[0]);
	FPRINTF(stderr,"\n");

	starpu_data_partition_clean(handle, 1, var_handle);

	/* Unpartition the data, unregister it from StarPU and shutdown */
	//starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr, "OUT 5-dim Array: \n");
	print_5dim_data(handle);
	starpu_data_unregister(handle);

	starpu_free_noflag(arr5d, NX*NY*NZ*NT*NG*sizeof(int));

	starpu_shutdown();
	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
