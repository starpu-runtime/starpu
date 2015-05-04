/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011, 2012, 2013, 2015  CNRS
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
#define NY    4
#define PARTS 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
        unsigned i, j;
        int *factor = (int *) cl_arg;

        /* length of the matrix */
        unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
        unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
        /* local copy of the matrix pointer */
        int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	FPRINTF(stderr, "applying factor %d\n", *factor);
        for(j=0; j<ny ; j++)
	{
                for(i=0; i<nx ; i++)
		{
                        FPRINTF(stderr, "%4d ", val[(j*ld)+i]);
                        val[(j*ld)+i] *= *factor;
		}
		FPRINTF(stderr,"\n");
        }
	FPRINTF(stderr,"\n");
}

struct starpu_codelet cl =
{
	.cpu_funcs = {cpu_func},
	.cpu_funcs_name = {"cpu_func"},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "matrix_scal"
};

void split_func(void *buffers[], void *cl_arg)
{
        unsigned i, j;
        int *factor = (int *) cl_arg;

        /* length of the matrix */
        unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
        unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);

        /* local copy of the matrix pointer */
        int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	FPRINTF(stderr, "splitting\n");
        for(j=0; j<ny ; j++)
	{
                for(i=0; i<nx ; i++)
		{
                        FPRINTF(stderr, "%4d ", val[(j*ld)+i]);
		}
                FPRINTF(stderr,"\n");
        }
	FPRINTF(stderr,"\n");

	starpu_data_handle_t submatrix = starpu_data_lookup(val);
        /* Partition the sub-matrix in PARTS sub-sub-matrices */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = PARTS
	};
	starpu_data_partition(submatrix, &f);

        /* Submit a task on each sub-vector */
	for (i=0; i<starpu_data_get_nb_children(submatrix); i++)
	{
                struct starpu_task *task = starpu_task_create();
		task->handles[0] = starpu_data_get_sub_data(submatrix, 1, i);
                task->cl = &cl;
                task->cl_arg = factor;
                task->cl_arg_size = sizeof(*factor);

		int ret = starpu_task_submit(task);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}
	//starpu_data_unpartition(submatrix, STARPU_MAIN_RAM);
}

int main(int argc, char **argv)
{
	unsigned j, n=1;
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
        struct starpu_codelet split_cl =
	{
                .cpu_funcs = {split_func},
                .cpu_funcs_name = {"split_func"},
                .nbuffers = 1,
		.modes = {STARPU_RW},
		.name = "split_matrix"
        };

        ret = starpu_init(NULL);
	if (ret == -ENODEV)
		return 77;
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
		starpu_data_handle_t subdata = starpu_data_get_sub_data(handle, 1, i);
		task->handles[0] = subdata;
                task->cl = &split_cl;
                task->cl_arg = &factor;
                task->cl_arg_size = sizeof(factor);

		ret = starpu_task_submit(task);
		if (ret == -ENODEV) goto enodev;
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
	}

	starpu_task_wait_for_all();
	for (i=0; i<starpu_data_get_nb_children(handle); i++)
	{
		starpu_data_handle_t subdata = starpu_data_get_sub_data(handle, 1, i);
		starpu_data_unpartition(subdata, STARPU_MAIN_RAM);
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
			if (matrix[(j*NX)+i] != n*12)
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
	starpu_shutdown();
	return 77;
}
