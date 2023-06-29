/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2010	    Mehdi Juhoor
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
 * This examplifies the use of the 4D matrix shadow filters: a source "matrix" of
 * NX*NY*NZ*NT elements (plus SHADOW wrap-around elements) is partitioned into
 * matrices with some shadowing, and these are copied into a destination
 * "matrix2" of
 * NRPARTSX*NPARTSY*NPARTSZ*NPARTST*((NX/NPARTSX+2*SHADOWX)*(NY/NPARTSY+2*SHADOWY)*(NZ/NPARTSZ+2*SHADOWZ)*(NT/NPARTST+2*SHADOWT))
 * elements, partitioned in the traditional way, thus showing how shadowing
 * shows up.
 */

#include <starpu.h>

/* Shadow width */
#define SHADOWX 2
#define SHADOWY 2
#define SHADOWZ 1
#define SHADOWT 1
#define NX    6
#define NY    6
#define NZ    2
#define NT    2
#define PARTSX 2
#define PARTSY 2
#define PARTSZ 2
#define PARTST 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

void cpu_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	/* length of the shadowed source matrix */
	unsigned ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
	unsigned ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
	unsigned ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
	unsigned x = STARPU_TENSOR_GET_NX(buffers[0]);
	unsigned y = STARPU_TENSOR_GET_NY(buffers[0]);
	unsigned z = STARPU_TENSOR_GET_NZ(buffers[0]);
	unsigned t = STARPU_TENSOR_GET_NT(buffers[0]);
	/* local copy of the shadowed source matrix pointer */
	int *val = (int *)STARPU_TENSOR_GET_PTR(buffers[0]);

	/* length of the destination matrix */
	unsigned ldy2 = STARPU_TENSOR_GET_LDY(buffers[1]);
	unsigned ldz2 = STARPU_TENSOR_GET_LDZ(buffers[1]);
	unsigned ldt2 = STARPU_TENSOR_GET_LDT(buffers[1]);
	unsigned x2 = STARPU_TENSOR_GET_NX(buffers[1]);
	unsigned y2 = STARPU_TENSOR_GET_NY(buffers[1]);
	unsigned z2 = STARPU_TENSOR_GET_NZ(buffers[1]);
	unsigned t2 = STARPU_TENSOR_GET_NT(buffers[1]);
	/* local copy of the destination matrix pointer */
	int *val2 = (int *)STARPU_TENSOR_GET_PTR(buffers[1]);

	unsigned i, j, k, l;

	/* If things go right, sizes should match */
	STARPU_ASSERT(x == x2);
	STARPU_ASSERT(y == y2);
	STARPU_ASSERT(z == z2);
	STARPU_ASSERT(t == t2);
	for (l = 0; l < t; l++)
		for (k = 0; k < z; k++)
			for (j = 0; j < y; j++)
				for (i = 0; i < x; i++)
					val2[l*ldt2+k*ldz2+j*ldy2+i] = val[l*ldt+k*ldz+j*ldy+i];
}

#ifdef STARPU_USE_CUDA
void cuda_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	/* length of the shadowed source matrix*/
	unsigned ldy = STARPU_TENSOR_GET_LDY(buffers[0]);
	unsigned ldz = STARPU_TENSOR_GET_LDZ(buffers[0]);
	unsigned ldt = STARPU_TENSOR_GET_LDT(buffers[0]);
	unsigned x = STARPU_TENSOR_GET_NX(buffers[0]);
	unsigned y = STARPU_TENSOR_GET_NY(buffers[0]);
	unsigned z = STARPU_TENSOR_GET_NZ(buffers[0]);
	unsigned t = STARPU_TENSOR_GET_NT(buffers[0]);
	/* local copy of the shadowed source matrix pointer */
	int *val = (int *)STARPU_TENSOR_GET_PTR(buffers[0]);

	/* length of the destination matrix */
	unsigned ldy2 = STARPU_TENSOR_GET_LDY(buffers[1]);
	unsigned ldz2 = STARPU_TENSOR_GET_LDZ(buffers[1]);
	unsigned ldt2 = STARPU_TENSOR_GET_LDT(buffers[1]);
	unsigned x2 = STARPU_TENSOR_GET_NX(buffers[1]);
	unsigned y2 = STARPU_TENSOR_GET_NY(buffers[1]);
	unsigned z2 = STARPU_TENSOR_GET_NZ(buffers[1]);
	unsigned t2 = STARPU_TENSOR_GET_NT(buffers[1]);
	/* local copy of the destination matrix pointer */
	int *val2 = (int *)STARPU_TENSOR_GET_PTR(buffers[1]);

	unsigned k, l;
	cudaError_t cures;

	/* If things go right, sizes should match */
	STARPU_ASSERT(x == x2);
	STARPU_ASSERT(y == y2);
	STARPU_ASSERT(z == z2);
	STARPU_ASSERT(t == t2);
	for (l = 0; l < t; l++)
	{
		for (k = 0; k < z; k++)
		{
			cures = cudaMemcpy2DAsync(val2+k*ldz2+l*ldt2, ldy2*sizeof(*val2), val+k*ldz+l*ldt, ldy*sizeof(*val),
						  x*sizeof(*val), y, cudaMemcpyDeviceToDevice, starpu_cuda_get_local_stream());
			STARPU_ASSERT(!cures);
		}
	}
}
#endif

int main(void)
{
	unsigned i, j, k, l, m, n, p, q;
	int matrix[NT + 2*SHADOWT][NZ + 2*SHADOWZ][NY + 2*SHADOWY][NX + 2*SHADOWX];
	int matrix2[NT + PARTST*2*SHADOWT][NZ + PARTSZ*2*SHADOWZ][NY + PARTSY*2*SHADOWY][NX + PARTSX*2*SHADOWX];
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
	for(l=1 ; l<=NT ; l++)
		for(k=1 ; k<=NZ ; k++)
			for(j=1 ; j<=NY ; j++)
				for(i=1 ; i<=NX ; i++)
					matrix[SHADOWT+l-1][SHADOWZ+k-1][SHADOWY+j-1][SHADOWX+i-1] = i+j+k+l;

	/*copy cubes*/
	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for (j = SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k][j][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l][k][j][SHADOWX+i];
				}
	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for(k=SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k][j+NY][i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l][k][SHADOWY+j][i];
				}
	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for(k=0 ; k<SHADOWZ ; k++)
			for(j=SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k+NZ][j][i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l][SHADOWZ+k][j][i];
				}
	for (l = 0 ; l<SHADOWT ; l++)
		for(k=SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for(j=SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k][j][i];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k][j][i];
				}

	/*copy planes*/
	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k][j+NY][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l][k][j+NY][SHADOWX+i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l][k][SHADOWY+j][i+NX];
					matrix[l][k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[l][k][SHADOWY+j][SHADOWX+i];
				}

	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for (k=0 ; k<SHADOWZ ; k++)
			for(j = SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k+NZ][j][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l][k+NZ][j][SHADOWX+i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l][SHADOWZ+k][j][i+NX];
					matrix[l][SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[l][SHADOWZ+k][j][SHADOWX+i];
				}

	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for (k=0 ; k<SHADOWZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k+NZ][j+NY][i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l][k+NZ][SHADOWY+j][i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l][SHADOWZ+k][j+NY][i];
					matrix[l][SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[l][SHADOWZ+k][SHADOWY+j][i];
				}

	for (l=0 ; l<SHADOWT ; l++)
		for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for(j = SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k][j][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l+NT][k][j][SHADOWX+i];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k][j][i+NX];
					matrix[SHADOWT+NT+l][k][j][SHADOWX+NX+i] = matrix[SHADOWT+l][k][j][SHADOWX+i];
				}

	for (l=0 ; l<SHADOWT ; l++)
		for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k][j+NY][i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l+NT][k][SHADOWY+j][i];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k][j+NY][i];
					matrix[SHADOWT+NT+l][k][SHADOWY+NY+j][i] = matrix[SHADOWT+l][k][SHADOWY+j][i];
				}

	for (l=0 ; l<SHADOWT ; l++)
		for(k=0 ; k<SHADOWZ ; k++)
			for (j = SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k+NZ][j][i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l+NT][SHADOWZ+k][j][i];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k+NZ][j][i];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][j][i] = matrix[SHADOWT+l][SHADOWZ+k][j][i];
				}

	/* Copy borders */
	for (l = SHADOWT ; l<SHADOWT+NT ; l++)
		for (k=0 ; k<SHADOWZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l][k+NZ][j+NY][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l][k+NZ][j+NY][SHADOWX+i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l][k+NZ][SHADOWY+j][i+NX];
					matrix[l][k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[l][k+NZ][SHADOWY+j][SHADOWX+i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l][SHADOWZ+k][j+NY][i+NX];
					matrix[l][SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[l][SHADOWZ+k][j+NY][SHADOWX+i];
					matrix[l][SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[l][SHADOWZ+k][SHADOWY+j][i+NX];
					matrix[l][SHADOWZ+NZ+k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[l][SHADOWZ+k][SHADOWY+j][SHADOWX+i];
				}
	for (l=0 ; l<SHADOWT ; l++)
		for (k = SHADOWZ ; k<SHADOWZ+NZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k][j+NY][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l+NT][k][j+NY][SHADOWX+i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l+NT][k][SHADOWY+j][i+NX];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k][j+NY][i+NX];
					matrix[l][k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[l+NT][k][SHADOWY+j][SHADOWX+i];
					matrix[SHADOWT+NT+l][k][j][SHADOWX+NX+i] = matrix[SHADOWT+l][k][j+NY][SHADOWX+i];
					matrix[SHADOWT+NT+l][k][SHADOWY+NY+j][i] = matrix[SHADOWT+l][k][SHADOWY+j][i+NX];
					matrix[SHADOWT+NT+l][k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[SHADOWT+l][k][SHADOWY+j][SHADOWX+i];
				}
	for (l=0 ; l<SHADOWT ; l++)
		for(k=0 ; k<SHADOWZ ; k++)
			for (j = SHADOWY ; j<SHADOWY+NY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k+NZ][j][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l+NT][k+NZ][j][SHADOWX+i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l+NT][SHADOWZ+k][j][i+NX];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k+NZ][j][i+NX];
					matrix[l][SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[l+NT][SHADOWZ+k][j][SHADOWX+i];
					matrix[SHADOWT+NT+l][k][j][SHADOWX+NX+i] = matrix[SHADOWT+l][k+NZ][j][SHADOWX+i];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][j][i] = matrix[SHADOWT+l][SHADOWZ+k][j][i+NX];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[SHADOWT+l][SHADOWZ+k][j][SHADOWX+i];
				}
	for (l=0 ; l<SHADOWT ; l++)
		for(k=0 ; k<SHADOWZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=SHADOWX ; i<SHADOWX+NX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k+NZ][j+NY][i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l+NT][k+NZ][SHADOWY+j][i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l+NT][SHADOWZ+k][j+NY][i];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k+NZ][j+NY][i];
					matrix[l][SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[l+NT][SHADOWZ+k][SHADOWY+j][i];
					matrix[SHADOWT+NT+l][k][SHADOWY+NY+j][i] = matrix[SHADOWT+l][k+NZ][SHADOWY+j][i];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][j][i] = matrix[SHADOWT+l][SHADOWZ+k][j+NY][i];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[SHADOWT+l][SHADOWZ+k][SHADOWY+j][i];
				}

	/* Copy corners */
	for(l=0 ; l<SHADOWT ; l++)
		for(k=0 ; k<SHADOWZ ; k++)
			for(j=0 ; j<SHADOWY ; j++)
				for(i=0 ; i<SHADOWX ; i++)
				{
					matrix[l][k][j][i] = matrix[l+NT][k+NZ][j+NY][i+NX];
					matrix[l][k][j][SHADOWX+NX+i] = matrix[l+NT][k+NZ][j+NY][SHADOWX+i];
					matrix[l][k][SHADOWY+NY+j][i] = matrix[l+NT][k+NZ][SHADOWY+j][i+NX];
					matrix[l][k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[l+NT][k+NZ][SHADOWY+j][SHADOWX+i];
					matrix[l][SHADOWZ+NZ+k][j][i] = matrix[l+NT][SHADOWZ+k][j+NY][i+NX];
					matrix[l][SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[l+NT][SHADOWZ+k][j+NY][SHADOWX+i];
					matrix[l][SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[l+NT][SHADOWZ+k][SHADOWY+j][i+NX];
					matrix[l][SHADOWZ+NZ+k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[l+NT][SHADOWZ+k][SHADOWY+j][SHADOWX+i];
					matrix[SHADOWT+NT+l][k][j][i] = matrix[SHADOWT+l][k+NZ][j+NY][i+NX];
					matrix[SHADOWT+NT+l][k][j][SHADOWX+NX+i] = matrix[SHADOWT+l][k+NZ][j+NY][SHADOWX+i];
					matrix[SHADOWT+NT+l][k][SHADOWY+NY+j][i] = matrix[SHADOWT+l][k+NZ][SHADOWY+j][i+NX];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][j][i] = matrix[SHADOWT+l][SHADOWZ+k][j+NY][i+NX];
					matrix[SHADOWT+NT+l][k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[SHADOWT+l][k+NZ][SHADOWY+j][SHADOWX+i];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][j][SHADOWX+NX+i] = matrix[SHADOWT+l][SHADOWZ+k][j+NY][SHADOWX+i];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][SHADOWY+NY+j][i] = matrix[SHADOWT+l][SHADOWZ+k][SHADOWY+j][i+NX];
					matrix[SHADOWT+NT+l][SHADOWZ+NZ+k][SHADOWY+NY+j][SHADOWX+NX+i] = matrix[SHADOWT+l][SHADOWZ+k][SHADOWY+j][SHADOWX+i];
				}

	FPRINTF(stderr,"IN  Matrix:\n");
	for(l=0 ; l<NT + 2*SHADOWT ; l++)
	{
		for(k=0 ; k<NZ + 2*SHADOWZ ; k++)
		{
			for(j=0 ; j<NY + 2*SHADOWY ; j++)
			{
				for(i=0 ; i<NX + 2*SHADOWX ; i++)
					FPRINTF(stderr, "%5d ", matrix[l][k][j][i]);
				FPRINTF(stderr,"\n");
			}
			FPRINTF(stderr,"\n\n");
		}
		FPRINTF(stderr,"\n\n");
	}
	FPRINTF(stderr,"\n");

	ret = starpu_init(NULL);
	if (ret == -ENODEV)
		exit(77);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* Declare source matrix to StarPU */
	starpu_tensor_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)matrix,
				    NX + 2*SHADOWX, (NX + 2*SHADOWX) * (NY + 2*SHADOWY), (NX + 2*SHADOWX) * (NY + 2*SHADOWY) * (NZ + 2*SHADOWZ),
				    NX + 2*SHADOWX, NY + 2*SHADOWY, NZ + 2*SHADOWZ, NT + 2*SHADOWT,
				    sizeof(matrix[0][0][0][0]));

	/* Declare destination matrix to StarPU */
	starpu_tensor_data_register(&handle2, STARPU_MAIN_RAM, (uintptr_t)matrix2,
				    NX + PARTSX*2*SHADOWX, (NX + PARTSX*2*SHADOWX) * (NY + PARTSY*2*SHADOWY), (NX + PARTSX*2*SHADOWX) * (NY + PARTSY*2*SHADOWY) * (NZ + PARTSZ*2*SHADOWZ),
				    NX + PARTSX*2*SHADOWX, NY + PARTSY*2*SHADOWY, NZ + PARTSZ*2*SHADOWZ, NT + PARTST*2*SHADOWT,
				    sizeof(matrix2[0][0][0][0]));

	/* Partition the source matrix in PARTST*PARTSZ*PARTSY*PARTSX sub-matrices with shadows */
	/* NOTE: the resulting handles should only be used in read-only mode,
	 * as StarPU will not know how the overlapping parts would have to be
	 * combined. */
	struct starpu_data_filter ft =
	{
		.filter_func = starpu_tensor_filter_time_block_shadow,
		.nchildren = PARTST,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWT /* Shadow width */
	};
	struct starpu_data_filter fz =
	{
		.filter_func = starpu_tensor_filter_depth_block_shadow,
		.nchildren = PARTSZ,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWZ /* Shadow width */
	};
	struct starpu_data_filter fy =
	{
		.filter_func = starpu_tensor_filter_vertical_block_shadow,
		.nchildren = PARTSY,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWY /* Shadow width */
	};
	struct starpu_data_filter fx =
	{
		.filter_func = starpu_tensor_filter_block_shadow,
		.nchildren = PARTSX,
		.filter_arg_ptr = (void*)(uintptr_t) SHADOWX /* Shadow width */
	};
	starpu_data_map_filters(handle, 4, &ft, &fz, &fy, &fx);

	/* Partition the destination matrix in PARTST*PARTSZ*PARTSY*PARTSX sub-matrices */
	struct starpu_data_filter ft2 =
	{
		.filter_func = starpu_tensor_filter_time_block,
		.nchildren = PARTST,
	};
	struct starpu_data_filter fz2 =
	{
		.filter_func = starpu_tensor_filter_depth_block,
		.nchildren = PARTSZ,
	};
	struct starpu_data_filter fy2 =
	{
		.filter_func = starpu_tensor_filter_vertical_block,
		.nchildren = PARTSY,
	};
	struct starpu_data_filter fx2 =
	{
		.filter_func = starpu_tensor_filter_block,
		.nchildren = PARTSX,
	};
	starpu_data_map_filters(handle2, 4, &ft2, &fz2, &fy2, &fx2);

	/* Submit a task on each sub-matrix */
	for (l=0; l<PARTST; l++)
	{
		for (k=0; k<PARTSZ; k++)
		{
			for (j=0; j<PARTSY; j++)
			{
				for (i=0; i<PARTSX; i++)
				{
					starpu_data_handle_t sub_handle = starpu_data_get_sub_data(handle, 4, l, k, j, i);
					starpu_data_handle_t sub_handle2 = starpu_data_get_sub_data(handle2, 4, l, k, j, i);
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
	}

	starpu_data_unpartition(handle, STARPU_MAIN_RAM);
	starpu_data_unpartition(handle2, STARPU_MAIN_RAM);
	starpu_data_unregister(handle);
	starpu_data_unregister(handle2);
	starpu_shutdown();

	FPRINTF(stderr,"OUT Matrix:\n");
	for(l=0 ; l<NT + PARTST*2*SHADOWT ; l++)
	{
		for(k=0 ; k<NZ + PARTSZ*2*SHADOWZ ; k++)
		{
			for(j=0 ; j<NY + PARTSY*2*SHADOWY ; j++)
			{
				for(i=0 ; i<NX + PARTSX*2*SHADOWX ; i++)
				{
					FPRINTF(stderr, "%5d ", matrix2[l][k][j][i]);
				}
				FPRINTF(stderr,"\n");
			}
			FPRINTF(stderr,"\n\n");
		}
		FPRINTF(stderr,"\n\n");
	}
	FPRINTF(stderr,"\n");
	for(l=0 ; l<PARTST ; l++)
		for(k=0 ; k<PARTSZ ; k++)
			for(j=0 ; j<PARTSY ; j++)
				for(i=0 ; i<PARTSX ; i++)
					for (q=0 ; q<NT/PARTST + 2*SHADOWT ; q++)
						for (p=0 ; p<NZ/PARTSZ + 2*SHADOWZ ; p++)
							for (n=0 ; n<NY/PARTSY + 2*SHADOWY ; n++)
								for (m=0 ; m<NX/PARTSX + 2*SHADOWX ; m++)
									STARPU_ASSERT(matrix2[l*(NT/PARTST+2*SHADOWT)+q][k*(NZ/PARTSZ+2*SHADOWZ)+p][j*(NY/PARTSY+2*SHADOWY)+n][i*(NX/PARTSX+2*SHADOWX)+m] ==
										      matrix[l*(NT/PARTST)+q][k*(NZ/PARTSZ)+p][j*(NY/PARTSY)+n][i*(NX/PARTSX)+m]);

	return 0;

enodev:
	FPRINTF(stderr, "WARNING: No one can execute this task\n");
	starpu_shutdown();
	return 77;
}

