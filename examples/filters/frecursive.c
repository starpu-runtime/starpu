/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_codelet(void *buffers[], void *cl_arg)
{
        unsigned i, j;
        int factor;

	starpu_codelet_unpack_args(cl_arg, &factor, 0);
        /* length of the matrix */
        unsigned nx = STARPU_MATRIX_GET_NX(buffers[0]);
        unsigned ny = STARPU_MATRIX_GET_NY(buffers[0]);
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
        /* local copy of the matrix pointer */
        int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

	FPRINTF(stderr, "computing on matrix with nx=%u, ny=%u, ld=%u\n", nx, ny, ld);
        for(j=0; j<ny ; j++)
	{
                for(i=0; i<nx ; i++)
                        val[(j*ld)+i] *= factor;
        }
}

static struct starpu_codelet cl =
{
        .cpu_funcs[0] = cpu_codelet,
        .nbuffers = 1,
	.modes[0] = STARPU_RW,
};

#define NX 400
#define NY 80
#define LD NX
#define PARTS 4

int main(void)
{
        int *matrix;
	starpu_data_handle_t matrix_handle;
	starpu_data_handle_t subhandle_l1[PARTS];
	starpu_data_handle_t subhandle_l2[PARTS][PARTS];
	starpu_data_handle_t subhandle_l3[PARTS][PARTS][PARTS];
	int ret;

	int factor = 12;
	int n=1;
	int i,j,k;

        ret = starpu_init(NULL);
	if (STARPU_UNLIKELY(ret == -ENODEV))
	{
		return 77;
	}

	if (starpu_cpu_worker_get_count() < 1)
	{
		FPRINTF(stderr, "This application requires at least 1 cpu worker\n");
		starpu_shutdown();
		return 77;
	}

	matrix = (int*)malloc(NX * NY * sizeof(int));
        assert(matrix);
	starpu_matrix_data_register(&matrix_handle, STARPU_MAIN_RAM, (uintptr_t)matrix, LD, NX, NY, sizeof(int));

        for(j=0 ; j<NY ; j++)
	{
                for(i=0 ; i<NX ; i++)
		{
                        matrix[(j*LD)+i] = n++;
                }
        }

	/* Split the matrix in PARTS sub-matrices, each sub-matrix in PARTS sub-sub-matrices, and each sub-sub matrix in PARTS sub-sub-sub-matrices */
	struct starpu_data_filter f =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = PARTS
	};
	struct starpu_data_filter f2 =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = PARTS
	};
	starpu_data_partition_plan(matrix_handle, &f, subhandle_l1);
	for(i=0 ; i<PARTS ; i++)
	{
		starpu_data_partition_plan(subhandle_l1[i], &f2, subhandle_l2[i]);
		for(j=0 ; j<PARTS ; j++)
		{
			starpu_data_partition_plan(subhandle_l2[i][j], &f, subhandle_l3[i][j]);
		}
	}

        /* Submit a task on the first sub-matrix and sub-sub matrix, and on all others sub-sub-matrices */
	ret = starpu_task_insert(&cl,
				 STARPU_RW, subhandle_l1[0],
				 STARPU_VALUE, &factor, sizeof(factor),
				 0);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	for (i=1; i<PARTS; i++)
	{
		ret = starpu_task_insert(&cl,
					 STARPU_RW, subhandle_l2[i][0],
					 STARPU_VALUE, &factor, sizeof(factor),
					 0);
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
		for (j=1; j<PARTS; j++)
		{
			for (k=0; k<PARTS; k++)
			{
				ret = starpu_task_insert(&cl,
							 STARPU_RW, subhandle_l3[i][j][k],
							 STARPU_VALUE, &factor, sizeof(factor),
							 0);
				STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
			}
		}
	}

	for(i=0 ; i<PARTS ; i++)
	{
		for(j=0 ; j<PARTS ; j++)
		{
			starpu_data_partition_clean(subhandle_l2[i][j], PARTS, subhandle_l3[i][j]);

		}
		starpu_data_partition_clean(subhandle_l1[i], PARTS, subhandle_l2[i]);
	}
	starpu_data_partition_clean(matrix_handle, PARTS, subhandle_l1);
	starpu_data_unregister(matrix_handle);

	/* Print result matrix */
	n=1;
	for(j=0 ; j<NY ; j++)
	{
		for(i=0 ; i<NX ; i++)
		{
			if (matrix[(j*LD)+i] != (int) n*12)
			{
				FPRINTF(stderr, "Incorrect result %4d != %4d", matrix[(j*LD)+i], n*12);
				ret=1;
			}
			n++;
		}
	}

	free(matrix);
        starpu_shutdown();

	return ret;
}
