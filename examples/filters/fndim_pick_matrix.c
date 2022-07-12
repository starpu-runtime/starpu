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

#define NX    5
#define NY    4
#define NZ    3
#define PARTS 2
#define POS   1

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void matrix_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void matrix_cuda_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_HIP
extern void matrix_hip_func(void *buffers[], void *cl_arg);
#endif

extern void generate_block_data(int *block, int nx, int ny, int nz, unsigned ldy, unsigned ldz);
extern void print_3dim_data(starpu_data_handle_t ndim_handle);
extern void print_matrix_data(starpu_data_handle_t matrix_handle);

int main(void)
{
	int *arr3d;
	int i, j, k;
	int ret;
	int factor = 2;

	starpu_data_handle_t handle;
	struct starpu_codelet cl =
	{
		.cpu_funcs = {matrix_cpu_func},
		.cpu_funcs_name = {"matrix_cpu_func"},
#ifdef STARPU_USE_CUDA
		.cuda_funcs = {matrix_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
		.hip_funcs = {matrix_hip_func},
		.hip_flags = {STARPU_HIP_ASYNC},
#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "arr3d_pick_matrix_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&arr3d, NX*NY*NZ*sizeof(int));
	assert(arr3d);
	generate_block_data(arr3d, NX, NY, NZ, NX, NX*NY);

	unsigned nn[3] = {NX, NY, NZ};
	unsigned ldn[3] = {1, NX, NX*NY};

	/* Declare data to StarPU */
	starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)arr3d, ldn, nn, 3, sizeof(int));
	FPRINTF(stderr, "IN 3-dim Array: \n");
	print_3dim_data(handle);

	/* Partition the 3-dim array in PARTS sub-matrices */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_ndim_filter_pick_matrix,
		.filter_arg = 1, //Partition the array along Y dimension
		.filter_arg_ptr = (void*)(uintptr_t) POS,
		.nchildren = PARTS,
		/* the children use a matrix interface*/
		.get_child_ops = starpu_ndim_filter_pick_matrix_child_ops
	};
	starpu_data_partition(handle, &f);

	FPRINTF(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

	for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
	{
		starpu_data_handle_t matrix_handle = starpu_data_get_sub_data(handle, 1, i);
		FPRINTF(stderr, "Sub Matrix %d: \n", i);
		print_matrix_data(matrix_handle);

		/* Submit a task on each sub-matrix */
		struct starpu_task *task = starpu_task_create();

		FPRINTF(stderr,"Dealing with sub-matrix %d\n", i);
		task->cl = &cl;
		task->synchronous = 1;
		task->callback_func = NULL;
		task->handles[0] = matrix_handle;
		task->cl_arg = &factor;
		task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		/* Print result matrix */
		FPRINTF(stderr, "OUT Matrix %d: \n", i);
		print_matrix_data(matrix_handle);
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr, "OUT 3-dim Array: \n");
	print_3dim_data(handle);
	starpu_data_unregister(handle);

	starpu_free_noflag(arr3d, NX*NY*NZ*sizeof(int));

	starpu_shutdown();
	return 0;

enodev:
	starpu_shutdown();
	return 77;
}
