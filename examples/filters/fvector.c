/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This examplifies how to use partitioning filters.  We here just split a
 * vector into slices, and run a dumb kernel on them.
 */

#include <starpu.h>

#define NX    21
#define PARTS 3

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void vector_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void vector_cuda_func(void *buffers[], void *cl_arg);
#endif
#ifdef STARPU_USE_HIP
extern void vector_hip_func(void *buffers[], void *cl_arg);
#endif

int main(void)
{
	int i;
	int* vector;
	starpu_data_handle_t handle;
	int factor=1;
	int ret;

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
		.name = "vector_scal"
	};

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_malloc((void **)&vector, NX*sizeof(int));
	for(i=0 ; i<NX ; i++) vector[i] = i;
	FPRINTF(stderr,"IN	Vector: ");
	for(i=0 ; i<NX ; i++) FPRINTF(stderr, "%5d ", vector[i]);
	FPRINTF(stderr,"\n");

	/* Declare data to StarPU */
	starpu_vector_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)vector, NX, sizeof(vector[0]));

	/* Partition the vector in PARTS sub-vectors */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_vector_filter_block,
		.nchildren = PARTS
	};
	starpu_data_partition(handle, &f);

	/* Submit a task on each sub-vector */
	for (i=0; i<starpu_data_get_nb_children(handle); i++)
	{
		starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 1, i);
		struct starpu_task *task = starpu_task_create();

		factor *= 10;
		task->handles[0] = sub_handle;
		task->cl = &cl;
		task->synchronous = 1;
		task->cl_arg = &factor;
		task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	/* Unpartition the data, unregister it from StarPU and shutdown */
	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unregister(handle);

	FPRINTF(stderr,"OUT Vector: ");
	for(i=0 ; i<NX ; i++) FPRINTF(stderr, "%5d ", vector[i]);
	FPRINTF(stderr,"\n");

	starpu_free_noflag(vector, NX*sizeof(int));
	starpu_shutdown();

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
