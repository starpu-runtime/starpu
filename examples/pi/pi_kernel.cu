/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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

#include "pi.h"

#define BLOCK_SIZE	256

static __global__ void monte_carlo(TYPE *random_numbers_x, TYPE *random_numbers_y,
						unsigned n, unsigned *output_cnt)
{
	__shared__ unsigned scnt[BLOCK_SIZE];

	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	/* Blank the shared mem buffer */
	if (threadIdx.x < 32)
		scnt[threadIdx.x] = 0;

	__syncthreads();

	/* Do we have a successful shot ? */
	TYPE x = random_numbers_x[tid];
	TYPE y = random_numbers_y[tid];
	TYPE dist = (x*x + y*y);

	scnt[threadIdx.x] = (dist <= 1.0f);

	__syncthreads();

	/* XXX that's totally unoptimized : we should do a reduction ! */
	if (threadIdx.x == 0)
	{
		unsigned total_cnt = 0;
		unsigned i;
		for (i = 0; i < BLOCK_SIZE; i++)
			total_cnt += scnt[i];

		output_cnt[blockIdx.x] = total_cnt;
	}
}

static __global__ void sum_per_block_cnt(unsigned previous_nblocks,
						unsigned *output_cnt,
						unsigned *cnt)
{
	/* XXX that's totally unoptimized yet : we should do a reduction ! */
	if (threadIdx.x == 0)
	{
		unsigned total_cnt = 0;
		unsigned i;
		for (i = 0; i < previous_nblocks; i++)
			total_cnt += output_cnt[i];

		*cnt = total_cnt;
	}
}

extern "C" void cuda_kernel(void *descr[], void *cl_arg)
{
	TYPE *random_numbers_x = (TYPE *)STARPU_GET_VECTOR_PTR(descr[0]);
	TYPE *random_numbers_y = (TYPE *)STARPU_GET_VECTOR_PTR(descr[1]);
	unsigned nx = STARPU_GET_VECTOR_NX(descr[0]);

	unsigned *cnt = (unsigned *)STARPU_GET_VECTOR_PTR(descr[2]);
	
	unsigned *per_block_cnt;
	cudaMalloc((void **)&per_block_cnt, (nx/BLOCK_SIZE)*sizeof(unsigned));

	monte_carlo<<<nx/BLOCK_SIZE, BLOCK_SIZE>>>(random_numbers_x, random_numbers_y, nx, per_block_cnt);
	cudaThreadSynchronize();

	sum_per_block_cnt<<<1, 32>>>(nx/BLOCK_SIZE, per_block_cnt, cnt);
	cudaThreadSynchronize();

	cudaFree(per_block_cnt);
}
