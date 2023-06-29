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

/*
 * This examplifies how to use partitioning filters.  We here just split a 3D
 * matrix into 3D slices (along the X axis), and run a dumb kernel on them.
 */

#include <starpu.h>

#define NX    5
#define NY    4
#define NZ    3
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void block_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void block_cuda_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_HIP
extern void block_hip_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_OPENCL
extern void opencl_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

extern void generate_block_data(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz);
extern void print_block(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz);
extern void print_block_data(starpu_data_handle_t block_handle);

int main(void)
{
	int *block;
	int i, j, k;
	int ret;

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
#ifdef STARPU_USE_OPENCL
		.opencl_funcs = {opencl_func},
		.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "block_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&block, NX*NY*NZ*sizeof(int));
	assert(block);
	generate_block_data(block, NX, NY, NZ, NX, NX*NY);

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_load_opencl_from_file("examples/filters/fblock_opencl_kernel.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif

	/* Declare data to StarPU */
	starpu_block_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)block, NX, NX*NY, NX, NY, NZ, sizeof(int));
	FPRINTF(stderr, "IN  Block\n");
	print_block_data(handle);

	/* Partition the block in PARTS sub-blocks */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_block_filter_block,
		.nchildren = PARTS
	};
	starpu_data_partition(handle, &f);

	FPRINTF(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

	for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
	{
		starpu_data_handle_t sblock = starpu_data_get_sub_data(handle, 1, i);
		FPRINTF(stderr, "Sub block %d\n", i);
		print_block_data(sblock);
	}

	/* Submit a task on each sub-block */
	for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
	{
		int multiplier=i;
		struct starpu_task *task = starpu_task_create();

		FPRINTF(stderr,"Dealing with sub-block %d\n", i);
		task->cl = &cl;
		task->synchronous = 1;
		task->callback_func = NULL;
		task->handles[0] = starpu_data_get_sub_data(handle, 1, i);
		task->cl_arg = &multiplier;
		task->cl_arg_size = sizeof(multiplier);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	print_block_data(handle);
	starpu_data_unregister(handle);

#ifdef STARPU_USE_OPENCL
	ret = starpu_opencl_unload_opencl(&opencl_program);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif

	/* Print result block */
	FPRINTF(stderr, "OUT Block\n");
	print_block(block, NX, NY, NZ, NX, NX*NY);

	starpu_free_noflag(block, NX*NY*NZ*sizeof(int));

	starpu_shutdown();
	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
