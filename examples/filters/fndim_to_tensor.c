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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void tensor_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void tensor_cuda_func(void *buffers[], void *cl_arg);
#endif

#ifdef STARPU_USE_HIP
extern void tensor_hip_func(void *buffers[], void *cl_arg);
#endif

extern void generate_tensor_data(int *tensor, int nx, int ny, int nz, int nt, unsigned ldy, unsigned ldz, unsigned ldt);
extern void print_4dim_data(starpu_data_handle_t ndim_handle);
extern void print_tensor_data(starpu_data_handle_t tensor_handle);

int main(void)
{
	int *arr4d;
	int i, j, k, l;
	int ret;

	arr4d = (int*)malloc(NX*NY*NZ*NT*sizeof(arr4d[0]));
	assert(arr4d);
	generate_tensor_data(arr4d, NX, NY, NZ, NT, NX, NX*NY, NX*NY*NZ);

	starpu_data_handle_t handle;
	struct starpu_codelet cl =
	{
		.cpu_funcs = {tensor_cpu_func},
		.cpu_funcs_name = {"tensor_cpu_func"},
#ifdef STARPU_USE_CUDA
		.cuda_funcs = {tensor_cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
		.hip_funcs = {tensor_hip_func},
		.hip_flags = {STARPU_HIP_ASYNC},
#endif
		.nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "arr4d_to_tensor_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	unsigned nn[4] = {NX, NY, NZ, NT};
	unsigned ldn[4] = {1, NX, NX*NY, NX*NY*NZ};

	/* Declare data to StarPU */
	starpu_ndim_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)arr4d, ldn, nn, 4, sizeof(int));
	FPRINTF(stderr, "IN 4-dim Array: \n");
	print_4dim_data(handle);

	/* Partition the 4-dim array in PARTS sub-tensors */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_ndim_filter_to_tensor,
		.filter_arg = 0, //Partition the array along X dimension
		.nchildren = PARTS,
		/* the children use a tensor interface*/
		.get_child_ops = starpu_ndim_filter_to_tensor_child_ops
	};
	starpu_data_partition(handle, &f);

	FPRINTF(stderr,"Nb of partitions : %d\n",starpu_data_get_nb_children(handle));

	for(i=0 ; i<starpu_data_get_nb_children(handle) ; i++)
	{
		starpu_data_handle_t tensor_handle = starpu_data_get_sub_data(handle, 1, i);
		FPRINTF(stderr, "Sub Tensor %d: \n", i);
		print_tensor_data(tensor_handle);

		/* Submit a task on each sub-tensor */
		int multiplier=i;
		struct starpu_task *task = starpu_task_create();

		FPRINTF(stderr,"Dealing with sub-tensor %d\n", i);
		task->cl = &cl;
		task->synchronous = 1;
		task->callback_func = NULL;
		task->handles[0] = tensor_handle;
		task->cl_arg = &multiplier;
		task->cl_arg_size = sizeof(multiplier);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");

		/* Print result tensor*/
		FPRINTF(stderr, "OUT Tensor %d: \n", i);
		print_tensor_data(tensor_handle);
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	FPRINTF(stderr, "OUT 4-dim Array: \n");
	print_4dim_data(handle);
	starpu_data_unregister(handle);

	free(arr4d);

	starpu_shutdown();
	return 0;

 enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
