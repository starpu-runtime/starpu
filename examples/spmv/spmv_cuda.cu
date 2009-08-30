/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <stdint.h>

#define MIN(a,b)	((a)<(b)?(a):(b))

extern "C" __global__ 
void spmv_kernel(uint32_t nnz, uint32_t nrow, float *nzval, uint32_t *colind, uint32_t *rowptr, 
			uint32_t firstentry, uint32_t elemsize, 
			float *vecin, uint32_t nx_in, uint32_t elemsize1, float * vecout, uint32_t nx_out, uint32_t elemsize2)
{
	/* only one dimension is used here */
	unsigned nthreads = gridDim.x*blockDim.x;
	unsigned threadid = threadIdx.x + blockIdx.x*blockDim.x;

	unsigned rowstart = threadid * ((nrow + (nthreads - 1))/nthreads);
	unsigned rowend = MIN(nrow, (threadid+1) * ((nrow + (nthreads - 1))/nthreads));

	unsigned row;
	for (row = rowstart; row < rowend; row++)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry; 

		for (index = firstindex; index < lastindex; index++)
		{
			tmp += nzval[index]*vecin[colind[index]];
		}

		vecout[row] = tmp;
	}
}

extern "C" __global__ 
void spmv_kernel_3(uint32_t nnz, uint32_t nrow, float *nzval, uint32_t *colind, uint32_t *rowptr, 
			uint32_t firstentry, 
			float *vecin, uint32_t nx_in, float * vecout, uint32_t nx_out)
{
	/* only one dimension is used here */
	unsigned block_rowstart = blockIdx.x*( (nrow + gridDim.x - 1)/gridDim.x );
	unsigned block_rowend = MIN((blockIdx.x+1)*( (nrow + gridDim.x - 1)/gridDim.x ), nrow);

	unsigned row;
	for (row = block_rowstart + threadIdx.x; row < block_rowend; row+=blockDim.x)
	{
		float tmp = 0.0f;
		unsigned index;

		unsigned firstindex = rowptr[row] - firstentry;
		unsigned lastindex = rowptr[row+1] - firstentry;

		for (index = firstindex; index < lastindex; index++)
		{
			tmp += nzval[index]*vecin[colind[index]];
		}

		vecout[row] = tmp;
	}
	

}

extern "C" void spmv_kernel_cpu_wrapper(uint32_t nnz, uint32_t nrow, float *nzval,
			uint32_t *colind, uint32_t *rowptr, uint32_t firstentry,
			float *vecin, uint32_t nx_in,
			float * vecout, uint32_t nx_out)
{
	dim3 dimBlock(8, 1);
	dim3 dimGrid(512, 1);

	spmv_kernel_3<<<dimGrid, dimBlock>>>(nnz, nrow, nzval, colind, rowptr,
						firstentry, vecin, nx_in, vecout, nx_out);

	cudaThreadSynchronize();
}

