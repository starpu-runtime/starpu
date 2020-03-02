/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * This examplifies the use of the matrix shadow filters: a source "matrix" of
 * NX*NY elements (plus 2*NX*SHADOWX+2*NY*SHADOWY+4*SHADOWX*SHADOWY wrap-around
 * elements) is partitioned into matrices with some shadowing, and these are
 * copied into a destination "matrix2" of
 * NRPARTSX*NPARTSY*((NX/NPARTSX+2*SHADOWX)*(NY/NPARTSY+2*SHADOWY)) elements,
 * partitioned in the traditionnal way, thus showing how shadowing shows up.
 *
 * For instance, with NX=NY=8, SHADOWX=SHADOWY=1, and NPARTSX=NPARTSY=4:
 *
 * matrix
 * 0123456789
 * 1234567890
 * 2345678901
 * 3456789012
 * 4567890123
 * 5678901234
 * 6789012345
 * 7890123456
 * 8901234567
 * 9012345678
 *
 * is partitioned into 4*4 pieces:
 *
 * 0123 2345 4567 6789
 * 1234 3456 5678 7890
 * 2345 4567 6789 8901
 * 3456 5678 7890 9012
 *
 * 2345 4567 6789 8901
 * 3456 5678 7890 9012
 * 4567 6789 8901 0123
 * 5678 7890 9012 1234
 *
 * 4567 6789 8901 0123
 * 5678 7890 9012 1234
 * 6789 8901 0123 2345
 * 7890 9012 1234 3456
 *
 * 6789 8901 0123 2345
 * 7890 9012 1234 3456
 * 8901 0123 2345 4567
 * 9012 1234 3456 5678
 *
 * which are copied into the 4*4 destination subparts of matrix2, thus getting in
 * the end:
 *
 * 0123234545676789
 * 1234345656787890
 * 2345456767898901
 * 3456567878909012
 * 2345456767898901
 * 3456567878909012
 * 4567678989010123
 * 5678789090121234
 * 4567678989010123
 * 5678789090121234
 * 6789890101232345
 * 7890901212343456
 * 6789890101232345
 * 7890901212343456
 * 8901012323454567
 * 9012123434565678
 */

#include <starpu.h>

/* Shadow width */
#define SHADOWX 3
#define SHADOWY 2
#define NX    20
#define NY    30
#define PARTSX 2
#define PARTSY 3

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
        /* length of the shadowed source matrix */
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
        unsigned n = STARPU_MATRIX_GET_NX(buffers[0]);
        unsigned m = STARPU_MATRIX_GET_NY(buffers[0]);
        /* local copy of the shadowed source matrix pointer */
        int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

        /* length of the destination matrix */
        unsigned ld2 = STARPU_MATRIX_GET_LD(buffers[1]);
        unsigned n2 = STARPU_MATRIX_GET_NX(buffers[1]);
        unsigned m2 = STARPU_MATRIX_GET_NY(buffers[1]);
        /* local copy of the destination matrix pointer */
        int *val2 = (int *)STARPU_MATRIX_GET_PTR(buffers[1]);

	unsigned i, j;

	/* If things go right, sizes should match */
	STARPU_ASSERT(n == n2);
	STARPU_ASSERT(m == m2);
	for (j = 0; j < m; j++)
		for (i = 0; i < n; i++)
			val2[j*ld2+i] = val[j*ld+i];
}

#ifdef STARPU_USE_CUDA
void cuda_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	cudaError_t cures;

        /* length of the shadowed source matrix */
        unsigned ld = STARPU_MATRIX_GET_LD(buffers[0]);
        unsigned n = STARPU_MATRIX_GET_NX(buffers[0]);
        unsigned m = STARPU_MATRIX_GET_NY(buffers[0]);
        /* local copy of the shadowed source matrix pointer */
        int *val = (int *)STARPU_MATRIX_GET_PTR(buffers[0]);

        /* length of the destination matrix */
        unsigned ld2 = STARPU_MATRIX_GET_LD(buffers[1]);
        unsigned n2 = STARPU_MATRIX_GET_NX(buffers[1]);
        unsigned m2 = STARPU_MATRIX_GET_NY(buffers[1]);
        /* local copy of the destination matrix pointer */
        int *val2 = (int *)STARPU_MATRIX_GET_PTR(buffers[1]);

	/* If things go right, sizes should match */
	STARPU_ASSERT(n == n2);
	STARPU_ASSERT(m == m2);
	cures = cudaMemcpy2DAsync(val2, ld2*sizeof(*val2), val, ld*sizeof(*val), n*sizeof(*val), m, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
        if (STARPU_UNLIKELY(cures)) STARPU_CUDA_REPORT_ERROR(cures);
}
#endif

int main(void)
{
	unsigned i, j, k, l;
        int matrix[NY + 2*SHADOWY][NX + 2*SHADOWX];
        int matrix2[NY + PARTSY*2*SHADOWY][NX + PARTSX*2*SHADOWX];
	starpu_data_handle_t handle, handle2;
	int ret;

        struct starpu_codelet cl =
	{
                .cpu_funcs = {cpu_func},
                .cpu_funcs_name = {"cpu_func"},
#ifdef STARPU_USE_CUDA
                .cuda_funcs = {cuda_func},
		.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
                .nbuffers = 2,
		.modes = {STARPU_R, STARPU_W}
        };

	memset(matrix, -1, sizeof(matrix));
	for(j=1 ; j<=NY ; j++)
		for(i=1 ; i<=NX ; i++)
			matrix[SHADOWY+j-1][SHADOWX+i-1] = i+j;

	/* Copy borders */
	for (j = SHADOWY ; j<SHADOWY+NY ; j++)
		for(i=0 ; i<SHADOWX ; i++)
		{
			matrix[j][i] = matrix[j][i+NX];
			matrix[j][SHADOWX+NX+i] = matrix[j][SHADOWX+i];
		}
	for(j=0 ; j<SHADOWY ; j++)
		for(i=SHADOWX ; i<SHADOWX+NX ; i++)
		{
			matrix[j][i] = matrix[j+NY][i];
			matrix[SHADOWY+NY+j][i] = matrix[SHADOWY+j][i];
		}
	/* Copy corners */
	for(j=0 ; j<SHADOWY ; j++)
		for(i=0 ; i<SHADOWX ; i++)
		{
			matrix[j][i] = matrix[j+NY][i+NX];
			matrix[j][SHADOWX+NX+i] = matrix[j+NY][SHADOWX+i];
			matrix[SHADOWY+NY+j][i] = matrix[SHADOWY+j][i+NX];
			matrix[SHADOWY+NY+j][SHADOWX+NX+i] = matrix[SHADOWY+j][SHADOWX+i];
		}

        FPRINTF(stderr,"IN  Matrix:\n");
	for(j=0 ; j<NY + 2*SHADOWY ; j++)
	{
		for(i=0 ; i<NX + 2*SHADOWX ; i++)
			FPRINTF(stderr, "%5d ", matrix[j][i]);
		FPRINTF(stderr,"\n");
	}
        FPRINTF(stderr,"\n");

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare source matrix to StarPU */
	starpu_matrix_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix, NX + 2*SHADOWX, NX + 2*SHADOWX, NY + 2*SHADOWY, sizeof(matrix[0][0]));

	/* Declare destination matrix to StarPU */
	starpu_matrix_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)matrix2, NX + PARTSX*2*SHADOWX, NX + PARTSX*2*SHADOWX, NY + PARTSY*2*SHADOWY, sizeof(matrix2[0][0]));

        /* Partition the source matrix in PARTSY*PARTSX sub-matrices with shadows */
	/* NOTE: the resulting handles should only be used in read-only mode,
	 * as StarPU will not know how the overlapping parts would have to be
	 * combined. */
	struct starpu_data_filter fy =
	{
		.filter_func = starpu_matrix_filter_vertical_block_shadow,
		.nchildren = PARTSY,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWY /* Shadow width */
	};
	struct starpu_data_filter fx =
	{
		.filter_func = starpu_matrix_filter_block_shadow,
		.nchildren = PARTSX,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWX /* Shadow width */
	};
	starpu_data_map_filters(handle, 2, &fy, &fx);

        /* Partition the destination matrix in PARTSY*PARTSX sub-matrices */
	struct starpu_data_filter fy2 =
	{
		.filter_func = starpu_matrix_filter_vertical_block,
		.nchildren = PARTSY,
	};
	struct starpu_data_filter fx2 =
	{
		.filter_func = starpu_matrix_filter_block,
		.nchildren = PARTSX,
	};
	starpu_data_map_filters(handle2, 2, &fy2, &fx2);

        /* Submit a task on each sub-matrix */
	for (j=0; j<PARTSY; j++)
	{
		for (i=0; i<PARTSX; i++)
		{
			starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 2, j, i);
			starpu_data_handle_t sub_handle2 = starpu_data_get_sub_data(handle2, 2, j, i);
			struct starpu_task *task = starpu_task_create();

			task->handles[0] = sub_handle;
			task->handles[1] = sub_handle2;
			task->cl = &cl;
			task->synchronous = 1;

			ret = starpu_task_submit(task);
			if (ret == -ENODEV) goto enodev;
			STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
		}
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(handle2, STARPU_MAIN_RAM);
        starpu_data_unregister(handle);
        starpu_data_unregister(handle2);
	starpu_shutdown();

        FPRINTF(stderr,"OUT Matrix:\n");
	for(j=0 ; j<NY + PARTSY*2*SHADOWY ; j++)
	{
		for(i=0 ; i<NX + PARTSX*2*SHADOWX ; i++)
			FPRINTF(stderr, "%5d ", matrix2[j][i]);
		FPRINTF(stderr,"\n");
	}
        FPRINTF(stderr,"\n");
	for(j=0 ; j<PARTSY ; j++)
		for(i=0 ; i<PARTSX ; i++)
			for (l=0 ; l<NY/PARTSY + 2*SHADOWY ; l++)
				for (k=0 ; k<NX/PARTSX + 2*SHADOWX ; k++)
					STARPU_ASSERT(matrix2[j*(NY/PARTSY+2*SHADOWY)+l][i*(NX/PARTSX+2*SHADOWX)+k] == matrix[j*(NY/PARTSY)+l][i*(NX/PARTSX)+k]);

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
