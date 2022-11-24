/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	int *factor = (int *) cl_arg;

	/* local copy of the variable pointer */
	int *val = (int *)STARPU_VARIABLE_GET_PTR(buffers[0]);

	*val *= *factor;
}

#ifdef STARPU_USE_CUDA
extern void variable_cuda_func(void *buffers[], void *cl_arg);
#endif

extern void generate_tensor_data(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt);
extern void print_tensor_data(starpu_data_handle_t tensor_handle);

int main(void)
{
	int *tensor;
	int i, j, k, l;
	int ret;
	int factor = 2;
	uint32_t pos[4] = {1,2,1,2};

	starpu_data_handle_t handle;
	starpu_data_handle_t var_handle[1];

	struct starpu_codelet cl =
	{
		.cpu_funcs = {cpu_func},
		.cpu_funcs_name = {"cpu_func"},
	#ifdef STARPU_USE_CUDA
		.cuda_funcs = {variable_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
	#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "tensor_pick_variable_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&tensor, NX*NY*NZ*NT*sizeof(int));
	assert(tensor);
	generate_tensor_data(tensor, NX, NY, NZ, NT, NX, NX*NY, NX*NY*NZ);

	/* Declare data to StarPU */
	starpu_tensor_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)tensor, NX, NX*NY, NX*NY*NZ, NX, NY, NZ, NT, sizeof(int));
	FPRINTF(stderr, "IN Tensor: \n");
	print_tensor_data(handle);

	/* Pick a variable in the tensor */
	struct starpu_data_filter f_var =
	{
		.filter_func = starpu_tensor_filter_pick_variable,
		.filter_arg_ptr = (void*)pos,
		.nchildren = 1,
		/* the children use a variable interface*/
		.get_child_ops = starpu_tensor_filter_pick_variable_child_ops
	};

	starpu_data_partition_plan(handle, &f_var, var_handle);

	FPRINTF(stderr, "Sub Variable:\n");
	int *variable = (int *)starpu_variable_get_local_ptr(var_handle[0]);
	FPRINTF(stderr, "%5d ", *variable);
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
	FPRINTF(stderr, "%5d ", *variable);
	FPRINTF(stderr,"\n");

	starpu_data_partition_clean(handle, 1, var_handle);

	/* Unpartition the data, unregister it from StarPU and shutdown */
	//starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr, "OUT Tensor: \n");
	print_tensor_data(handle);
	starpu_data_unregister(handle);

	starpu_free_noflag(tensor, NX*NY*NZ*NT*sizeof(int));

	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
