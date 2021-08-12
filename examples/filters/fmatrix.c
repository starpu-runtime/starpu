/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This examplifies how to use partitioning filters.  We here just split a 2D
 * matrix into 2D slices (along the X axis), and run a dumb kernel on them.
 */

#include <starpu.h>

#define NX    5
#define NY    4
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

extern void matrix_cpu_func(void *buffers[], void *cl_arg);

#ifdef STARPU_USE_CUDA
extern void matrix_cuda_func(void *buffers[], void *cl_arg);
#endif

int main(void)
{
	unsigned j;
	int n=1;
        int matrix[NX*NY];
	int ret, i;
	int factor = 12;

        FPRINTF(stderr,"IN  Matrix: \n");
        for(j=0 ; j<NY ; j++)
	{
                for(i=0 ; i<NX ; i++)
		{
                        matrix[(j*NX)+i] = n++;
                        FPRINTF(stderr, "%4d ", matrix[(j*NX)+i]);
                }
                FPRINTF(stderr,"\n");
        }
        FPRINTF(stderr,"\n");

        starpu_data_handle_t handle;
        struct starpu_codelet cl =
	{
                .cpu_funcs = {matrix_cpu_func},
                .cpu_funcs_name = {"matrix_cpu_func"},
#ifdef STARPU_USE_CUDA
                .cuda_funcs = {matrix_cuda_func},
        	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
                .nbuffers = 1,
        	.modes = {STARPU_RW},
        	.name = "matrix_scal"
        };

        ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare data to StarPU */
	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX, NX, NY, sizeof(matrix[0]));

        /* Partition the matrix in PARTS sub-matrices */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = PARTS
	};
	starpu_data_partition(handle, &f);

        /* Submit a task on each sub-vector */
	for (i=0; i<starpu_data_get_nb_children(handle); i++)
	{
                struct starpu_task *task = starpu_task_create();
		task->handles[0] = starpu_data_get_sub_data(handle, 1, i);
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
	starpu_shutdown();

        /* Print result matrix */
	n=1;
        FPRINTF(stderr,"OUT Matrix: \n");
        for(j=0 ; j<NY ; j++)
	{
                for(i=0 ; i<NX ; i++)
		{
                        FPRINTF(stderr, "%4d ", matrix[(j*NX)+i]);
			if (matrix[(j*NX)+i] != (int) n*12)
			{
				FPRINTF(stderr, "Incorrect result %4d != %4d", matrix[(j*NX)+i], n*12);
				ret=1;
			}
			n++;
                }
                FPRINTF(stderr,"\n");
        }
        FPRINTF(stderr,"\n");

	return ret;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
