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
#define PARTS 2
#define POS   2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void vector_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void vector_cuda_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_HIP
extern void vector_hip_func(void *buffers[], void *cl_arg);
#endif

extern void generate_matrix_data(int *matrix, int nx, int ny, unsigned ld);
extern void print_2dim_data(starpu_data_handle_t ndim_handle);

int main(void)
{
	int *arr2d;
	int ret, i, j;
	int factor = 12;

	arr2d = (int*)malloc(NX*NY*sizeof(int));
	generate_matrix_data(arr2d, NX, NY, NX);

	starpu_data_handle_t handle;
	struct starpu_codelet cl =
	{
		.cpu_funcs = {vector_cpu_func},
		.cpu_funcs_name = {"vector_cpu_func"},
#ifdef STARPU_USE_CUDA
		.cuda_funcs = {vector_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
		.hip_funcs = {vector_hip_func},
		.hip_flags = {STARPU_HIP_ASYNC},
#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "arr2d_pick_vector_scal"
        };

        ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned nn[2] = {NX, NY};
	unsigned ldn[2] = {1, NX};

	/* Declare data to StarPU */
	starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)arr2d, ldn, nn, 2, sizeof(int));
	FPRINTF(stderr, "IN 2-dim Array: \n");
	print_2dim_data(handle);

	/* Partition the 2-dim array in PARTS sub-vectors */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_ndim_filter_pick_vector,
		.filter_arg = 1, //Partition the array along Y dimension
		.filter_arg_ptr = (void*)(uintptr_t) POS,
		.nchildren = PARTS,
		/* the children use a vector interface*/
		.get_child_ops = starpu_ndim_filter_pick_vector_child_ops
	};
	starpu_data_partition(handle, &f);

	FPRINTF(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

	for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
	{
		starpu_data_handle_t vector_handle = starpu_data_get_sub_data(handle, 1, i);
		FPRINTF(stderr, "Sub Vector %d: \n", i);
		int *vector = (int *)starpu_vector_get_local_ptr(vector_handle);
		int nx = starpu_vector_get_nx(vector_handle);
		for(j=0 ; j<nx ; j++) FPRINTF(stderr, "%5d ", vector[j]);
		FPRINTF(stderr,"\n");
		
		/* Submit a task on each sub-vector */
		struct starpu_task *task = starpu_task_create();

		FPRINTF(stderr,"Dealing with sub-vector %d\n", i);
		task->handles[0] = vector_handle;
		task->cl = &cl;
		task->synchronous = 1;
		task->cl_arg = &factor;
		task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		/* Print result vector */
		FPRINTF(stderr,"OUT Vector %d: \n", i);
		for(j=0 ; j<nx ; j++) FPRINTF(stderr, "%5d ", vector[j]);
		FPRINTF(stderr,"\n");
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr,"OUT 2-dim Array: \n");
	print_2dim_data(handle);
	starpu_data_unregister(handle);

	free(arr2d);
	starpu_shutdown();

	return 0;

 enodev:
	starpu_shutdown();
	return 77;
}
