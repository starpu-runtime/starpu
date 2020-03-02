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
 * This examplifies the use of the 3D matrix shadow filters: a source "matrix" of
 * NX*NY*NZ elements (plus SHADOW wrap-around elements) is partitioned into
 * matrices with some shadowing, and these are copied into a destination
 * "matrix2" of
 * NRPARTSX*NPARTSY*NPARTSZ*((NX/NPARTSX+2*SHADOWX)*(NY/NPARTSY+2*SHADOWY)*(NZ/NPARTSZ+2*SHADOWZ))
 * elements, partitioned in the traditionnal way, thus showing how shadowing
 * shows up.
 */

#include <starpu.h>

/* Shadow width */
#define SHADOWX 2
#define SHADOWY 3
#define SHADOWZ 4
#define NX    12
#define NY    9
#define NZ    6
#define PARTSX 4
#define PARTSY 3
#define PARTSZ 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
        /* length of the shadowed source matrix */
        unsigned ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
        unsigned ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
        unsigned x = STARPU_BLOCK_GET_NX(buffers[0]);
        unsigned y = STARPU_BLOCK_GET_NY(buffers[0]);
        unsigned z = STARPU_BLOCK_GET_NZ(buffers[0]);
        /* local copy of the shadowed source matrix pointer */
        int *val = (int *)STARPU_BLOCK_GET_PTR(buffers[0]);

        /* length of the destination matrix */
        unsigned ldy2 = STARPU_BLOCK_GET_LDY(buffers[1]);
        unsigned ldz2 = STARPU_BLOCK_GET_LDZ(buffers[1]);
        unsigned x2 = STARPU_BLOCK_GET_NX(buffers[1]);
        unsigned y2 = STARPU_BLOCK_GET_NY(buffers[1]);
        unsigned z2 = STARPU_BLOCK_GET_NZ(buffers[1]);
        /* local copy of the destination matrix pointer */
        int *val2 = (int *)STARPU_BLOCK_GET_PTR(buffers[1]);

	unsigned i, j, k;

	/* If things go right, sizes should match */
	STARPU_ASSERT(x == x2);
	STARPU_ASSERT(y == y2);
	STARPU_ASSERT(z == z2);
	for (k = 0; k < z; k++)
		for (j = 0; j < y; j++)
			for (i = 0; i < x; i++)
				val2[k*ldz2+j*ldy2+i] = val[k*ldz+j*ldy+i];
}

#ifdef STARPU_USE_CUDA
void cuda_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
        /* length of the shadowed source matrix */
        unsigned ldy = STARPU_BLOCK_GET_LDY(buffers[0]);
        unsigned ldz = STARPU_BLOCK_GET_LDZ(buffers[0]);
        unsigned x = STARPU_BLOCK_GET_NX(buffers[0]);
        unsigned y = STARPU_BLOCK_GET_NY(buffers[0]);
        unsigned z = STARPU_BLOCK_GET_NZ(buffers[0]);
        /* local copy of the shadowed source matrix pointer */
        int *val = (int *)STARPU_BLOCK_GET_PTR(buffers[0]);

        /* length of the destination matrix */
        unsigned ldy2 = STARPU_BLOCK_GET_LDY(buffers[1]);
        unsigned ldz2 = STARPU_BLOCK_GET_LDZ(buffers[1]);
        unsigned x2 = STARPU_BLOCK_GET_NX(buffers[1]);
        unsigned y2 = STARPU_BLOCK_GET_NY(buffers[1]);
        unsigned z2 = STARPU_BLOCK_GET_NZ(buffers[1]);
        /* local copy of the destination matrix pointer */
        int *val2 = (int *)STARPU_BLOCK_GET_PTR(buffers[1]);

	unsigned k;
	cudaError_t cures;

	/* If things go right, sizes should match */
	STARPU_ASSERT(x == x2);
	STARPU_ASSERT(y == y2);
	STARPU_ASSERT(z == z2);
	for (k = 0; k < z; k++)
	{
		cures = cudaMemcpy2DAsync(val2+k*ldz2, ldy2*sizeof(*val2), val+k*ldz, ldy*sizeof(*val),
				x*sizeof(*val), y, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
		STARPU_ASSERT(!cures);
	}
}
#endif

int main(void)
{
	unsigned i, j, k, l, m, n;
        int matrix[NZ + 2*SHADOWZ][NY + 2*SHADOWY][NX + 2*SHADOWX];
        int matrix2[NZ + PARTSZ*2*SHADOWZ][NY + PARTSY*2*SHADOWY][NX + PARTSX*2*SHADOWX];
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
	for(k=1 ; k<=NZ ; k++)
		for(j=1 ; j<=NY ; j++)
			for(i=1 ; i<=NX ; i++)
				matrix[SHADOWZ+k-1][SHADOWY+j-1][SHADOWX+i-1] = i+j+k;

	/* Copy planes */
	for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
		for (j = SHADOWY ; j<SHADOWY+NY ; j++)
			for(i=0 ; i<SHADOWX ; i++)
			{
				matrix[k][j][i] = matrix[k][j][i+NX];
				matrix[k][j][SHADOWX+NX+i] = matrix[k][j][SHADOWX+i];
			}
	for(k=SHADOWZ ; k<SHADOWZ+NZ ; k++)
		for(j=0 ; j<SHADOWY ; j++)
			for(i=SHADOWX ; i<SHADOWX+NX ; i++)
			{
				matrix[k][j][i] = matrix[k][j+NY][i];
				matrix[k][SHADOWY+NY+j][i] = matrix[k][SHADOWY+j][i];
			}
	for(k=0 ; k<SHADOWZ ; k++)
		for(j=SHADOWY ; j<SHADOWY+NY ; j++)
			for(i=SHADOWX ; i<SHADOWX+NX ; i++)
			{
				matrix[k][j][i] = matrix[k+NZ][j][i];
				matrix[SHADOWZ+NZ+k][j][i] = matrix[SHADOWZ+k][j][i];
			}

	/* Copy borders */
	for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
		for(j=0 ; j<SHADOWY ; j++)
			for(i=0 ; i<SHADOWX ; i++)
			{
				matrix[k][j][i] = matrix[k][j+NY][i+NX];
				matrix[k][SHADOWY+NY+j][i] = matrix[k][SHADOWY+j][i+NX];
				matrix[k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[k][SHADOWY+j][SHADOWX+i];
				matrix[k][j][SHADOWX+NX+i] = matrix[k][j+NY][SHADOWX+i];
			}
	for(k=0 ; k<SHADOWZ ; k++)
		for (j = SHADOWY ; j<SHADOWY+NY ; j++)
			for(i=0 ; i<SHADOWX ; i++)
			{
				matrix[k][j][i] = matrix[k+NZ][j][i+NX];
				matrix[SHADOWZ+NZ+k][j][i] = matrix[SHADOWZ+k][j][i+NX];
				matrix[SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[SHADOWZ+k][j][SHADOWX+i];
				matrix[k][j][SHADOWX+NX+i] = matrix[k+NZ][j][SHADOWX+i];
			}
	for(k=0 ; k<SHADOWZ ; k++)
		for(j=0 ; j<SHADOWY ; j++)
			for(i=SHADOWX ; i<SHADOWX+NX ; i++)
			{
				matrix[k][j][i] = matrix[k+NZ][j+NY][i];
				matrix[SHADOWZ+NZ+k][j][i] = matrix[SHADOWZ+k][j+NY][i];
				matrix[SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[SHADOWZ+k][SHADOWY+j][i];
				matrix[k][SHADOWY+NY+j][i] = matrix[k+NZ][SHADOWY+j][i];
			}

	/* Copy corners */
	for(k=0 ; k<SHADOWZ ; k++)
		for(j=0 ; j<SHADOWY ; j++)
			for(i=0 ; i<SHADOWX ; i++)
			{
				matrix[k][j][i] = matrix[k+NZ][j+NY][i+NX];
				matrix[k][j][SHADOWX+NX+i] = matrix[k+NZ][j+NY][SHADOWX+i];
				matrix[k][SHADOWY+NY+j][i] = matrix[k+NZ][SHADOWY+j][i+NX];
				matrix[k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[k+NZ][SHADOWY+j][SHADOWX+i];
				matrix[SHADOWZ+NZ+k][j][i] = matrix[SHADOWZ+k][j+NY][i+NX];
				matrix[SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[SHADOWZ+k][j+NY][SHADOWX+i];
				matrix[SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[SHADOWZ+k][SHADOWY+j][i+NX];
				matrix[SHADOWZ+NZ+k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[SHADOWZ+k][SHADOWY+j][SHADOWX+i];
			}

        FPRINTF(stderr,"IN  Matrix:\n");
	for(k=0 ; k<NZ + 2*SHADOWZ ; k++)
	{
		for(j=0 ; j<NY + 2*SHADOWY ; j++)
		{
			for(i=0 ; i<NX + 2*SHADOWX ; i++)
				FPRINTF(stderr, "%5d ", matrix[k][j][i]);
			FPRINTF(stderr,"\n");
		}
		FPRINTF(stderr,"\n\n");
	}
        FPRINTF(stderr,"\n");

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare source matrix to StarPU */
	starpu_block_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix,
			NX + 2*SHADOWX, (NX + 2*SHADOWX) * (NY + 2*SHADOWY),
			NX + 2*SHADOWX, NY + 2*SHADOWY, NZ + 2*SHADOWZ,
			sizeof(matrix[0][0][0]));

	/* Declare destination matrix to StarPU */
	starpu_block_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)matrix2,
			NX + PARTSX*2*SHADOWX, (NX + PARTSX*2*SHADOWX) * (NY + PARTSY*2*SHADOWY),
			NX + PARTSX*2*SHADOWX, NY + PARTSY*2*SHADOWY, NZ + PARTSZ*2*SHADOWZ,
			sizeof(matrix2[0][0][0]));

        /* Partition the source matrix in PARTSZ*PARTSY*PARTSX sub-matrices with shadows */
	/* NOTE: the resulting handles should only be used in read-only mode,
	 * as StarPU will not know how the overlapping parts would have to be
	 * combined. */
	struct starpu_data_filter fz =
	{
		.filter_func = starpu_block_filter_depth_block_shadow,
		.nchildren = PARTSZ,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWZ /* Shadow width */
	};
	struct starpu_data_filter fy =
	{
		.filter_func = starpu_block_filter_vertical_block_shadow,
		.nchildren = PARTSY,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWY /* Shadow width */
	};
	struct starpu_data_filter fx =
	{
		.filter_func = starpu_block_filter_block_shadow,
		.nchildren = PARTSX,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWX /* Shadow width */
	};
	starpu_data_map_filters(handle, 3, &fz, &fy, &fx);

        /* Partition the destination matrix in PARTSZ*PARTSY*PARTSX sub-matrices */
	struct starpu_data_filter fz2 =
	{
		.filter_func = starpu_block_filter_depth_block,
		.nchildren = PARTSZ,
	};
	struct starpu_data_filter fy2 =
	{
		.filter_func = starpu_block_filter_vertical_block,
		.nchildren = PARTSY,
	};
	struct starpu_data_filter fx2 =
	{
		.filter_func = starpu_block_filter_block,
		.nchildren = PARTSX,
	};
	starpu_data_map_filters(handle2, 3, &fz2, &fy2, &fx2);

        /* Submit a task on each sub-matrix */
	for (k=0; k<PARTSZ; k++)
	{
		for (j=0; j<PARTSY; j++)
		{
			for (i=0; i<PARTSX; i++)
			{
				starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 3, k, j, i);
				starpu_data_handle_t sub_handle2 = starpu_data_get_sub_data(handle2, 3, k, j, i);
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
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(handle2, STARPU_MAIN_RAM);
        starpu_data_unregister(handle);
        starpu_data_unregister(handle2);
	starpu_shutdown();

        FPRINTF(stderr,"OUT Matrix:\n");
	for(k=0 ; k<NZ + PARTSZ*2*SHADOWZ ; k++)
	{
		for(j=0 ; j<NY + PARTSY*2*SHADOWY ; j++)
		{
			for(i=0 ; i<NX + PARTSX*2*SHADOWX ; i++)
			{
				FPRINTF(stderr, "%5d ", matrix2[k][j][i]);
			}
			FPRINTF(stderr,"\n");
		}
		FPRINTF(stderr,"\n\n");
	}
        FPRINTF(stderr,"\n");
	for(k=0 ; k<PARTSZ ; k++)
		for(j=0 ; j<PARTSY ; j++)
			for(i=0 ; i<PARTSX ; i++)
				for (n=0 ; n<NZ/PARTSZ + 2*SHADOWZ ; n++)
					for (m=0 ; m<NY/PARTSY + 2*SHADOWY ; m++)
						for (l=0 ; l<NX/PARTSX + 2*SHADOWX ; l++)
							STARPU_ASSERT(matrix2[k*(NZ/PARTSZ+2*SHADOWZ)+n][j*(NY/PARTSY+2*SHADOWY)+m][i*(NX/PARTSX+2*SHADOWX)+l] ==
									matrix[k*(NZ/PARTSZ)+n][j*(NY/PARTSY)+m][i*(NX/PARTSX)+l]);

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}
