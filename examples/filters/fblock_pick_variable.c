/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define NX    5
#define NY    4
#define NZ    3

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

extern void generate_block_data(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz);
extern void print_block_data(starpu_data_handle_t block_handle);

int main(void)
{
	int *block;
	int i, j, k;
	int ret;
	int factor = 2;
	uint32_t pos[3] = {1,2,1};

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
		.name = "block_pick_variable_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&block, NX*NY*NZ*sizeof(int));
	assert(block);
	generate_block_data(block, NX, NY, NZ, NX, NX*NY);

	/* Declare data to StarPU */
	starpu_block_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)block, NX, NX*NY, NX, NY, NZ, sizeof(int));
	FPRINTF(stderr, "IN Block: \n");
	print_block_data(handle);

	/* Pick a variable in the block */
	struct starpu_data_filter f_var =
	{
		.filter_func = starpu_block_filter_pick_variable,
		.filter_arg_ptr = (void*)pos,
		.nchildren = 1,
		/* the children use a variable interface*/
		.get_child_ops = starpu_block_filter_pick_variable_child_ops
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
	FPRINTF(stderr, "OUT Block: \n");
	print_block_data(handle);
	starpu_data_unregister(handle);

	starpu_free_noflag(block, NX*NY*NZ*sizeof(int));

	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
