/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#define PARTS 2
#define POS   1

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void block_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void block_cuda_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_HIP
extern void block_hip_func(void *buffers[], void *cl_arg);
#endif

extern void generate_tensor_data(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt);
extern void print_4dim_data(starpu_data_handle_t ndim_handle);
extern void print_block_data(starpu_data_handle_t block_handle);

int main(void)
{
	int *arr4d;
	int i, j, k, l;
	int ret;
	int factor = 2;

	starpu_data_handle_t handle;
	struct starpu_codelet cl =
	{
		.cpu_funcs = {block_cpu_func},
		.cpu_funcs_name = {"block_cpu_func"},
#ifdef STARPU_USE_CUDA
		.cuda_funcs = {block_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
		.hip_funcs = {block_hip_func},
		.hip_flags = {STARPU_HIP_ASYNC},
#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "arr4d_pick_block_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&arr4d, NX*NY*NZ*NT*sizeof(int));
	assert(arr4d);
	generate_tensor_data(arr4d, NX, NY, NZ, NT, NX, NX*NY, NX*NY*NZ);

	unsigned nn[4] = {NX, NY, NZ, NT};
	unsigned ldn[4] = {1, NX, NX*NY, NX*NY*NZ};

	/* Declare data to StarPU */
	starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)arr4d, ldn, nn, 4, sizeof(int));
	FPRINTF(stderr, "IN 4-dim Array: \n");
	print_4dim_data(handle);

	/* Partition the 4-dim array in PARTS sub-blocks */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_ndim_filter_pick_block,
		.filter_arg = 2, //Partition the array along Z dimension
		.filter_arg_ptr = (void*)(uintptr_t) POS,
		.nchildren = PARTS,
		/* the children use a block interface*/
		.get_child_ops = starpu_ndim_filter_pick_block_child_ops
	};
	starpu_data_partition(handle, &f);

	FPRINTF(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

	for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
	{
		starpu_data_handle_t block_handle = starpu_data_get_sub_data(handle, 1, i);
		FPRINTF(stderr, "Sub Block %d: \n", i);
		print_block_data(block_handle);

		/* Submit a task on each sub-block */
		struct starpu_task *task = starpu_task_create();

		FPRINTF(stderr,"Dealing with sub-block %d\n", i);
		task->cl = &cl;
		task->synchronous = 1;
		task->callback_func = NULL;
		task->handles[0] = block_handle;
		task->cl_arg = &factor;
		task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		/* Print result block */
		FPRINTF(stderr, "OUT Block %d: \n", i);
		print_block_data(block_handle);
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr, "OUT 4-dim Array: \n");
	print_4dim_data(handle);
	starpu_data_unregister(handle);

	starpu_free_noflag(arr4d, NX*NY*NZ*NT*sizeof(int));

	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
