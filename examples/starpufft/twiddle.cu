/*
 * StarPU
 * Copyright (C) INRIA 2009 (see AUTHORS file)
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

#include <cuComplex.h>

/* Note: these assume that the sizes are powers of two */

extern "C" __global__ void starpufftf_cuda_1d_twiddle(cuComplex * out, cuComplex * roots, unsigned n, unsigned i)
{
	unsigned j;
	unsigned start = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned end = start + 1;

	for (j = start; j < end; j++)
		out[j] = cuCmulf(out[j], roots[i*j]);
	return;
}

extern "C" void starpufftf_cuda_1d_twiddle_host(cuComplex *out, cuComplex *roots, unsigned n, unsigned i)
{
	unsigned threads_per_block = 128;

	if (n < threads_per_block) {
		dim3 dimGrid(n);
		starpufftf_cuda_1d_twiddle <<<dimGrid, 1>>> (out, roots, n, i);
	} else {
		dim3 dimGrid(n / threads_per_block);
		dim3 dimBlock(threads_per_block);
		starpufftf_cuda_1d_twiddle <<<dimGrid, dimBlock>>> (out, roots, n, i);
	}
}

extern "C" __global__ void starpufftf_cuda_2d_twiddle(cuComplex * out, cuComplex * roots0, cuComplex * roots1, unsigned n2, unsigned m2, unsigned i, unsigned j)
{
	unsigned k, l;
	unsigned startx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned starty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned endx = startx + 1;
	unsigned endy = starty + 1;

	//for (k = startx; k < endx ; k++)
		//for (l = starty; l < endy ; l++)
	k = startx;
	l = starty;
			out[k*m2 + l] = cuCmulf(cuCmulf(out[k*m2 + l], roots0[i*k]), roots1[j*l]);
	return;
}

extern "C" void starpufftf_cuda_2d_twiddle_host(cuComplex *out, cuComplex *roots0, cuComplex *roots1, unsigned n2, unsigned m2, unsigned i, unsigned j)
{
	unsigned threads_per_dim = 16;
	if (n2 < threads_per_dim) {
		if (m2 < threads_per_dim) {
			dim3 dimGrid(n2, m2);
			starpufftf_cuda_2d_twiddle <<<dimGrid, 1>>> (out, roots0, roots1, n2, m2, i, j);
		} else {
			dim3 dimGrid(n2, threads_per_dim);
			dim3 dimBlock(1, m2 / threads_per_dim);
			starpufftf_cuda_2d_twiddle <<<dimGrid, dimBlock>>> (out, roots0, roots1, n2, m2, i, j);
		}
	} else { 
		if (m2 < threads_per_dim) {
			dim3 dimGrid(threads_per_dim, m2);
			dim3 dimBlock(n2 / threads_per_dim, 1);
			starpufftf_cuda_2d_twiddle <<<dimGrid, dimBlock>>> (out, roots0, roots1, n2, m2, i, j);
		} else {
			dim3 dimGrid(threads_per_dim, threads_per_dim);
			dim3 dimBlock(n2 / threads_per_dim, m2 / threads_per_dim);
			starpufftf_cuda_2d_twiddle <<<dimGrid, dimBlock>>> (out, roots0, roots1, n2, m2, i, j);
		}
	}
}
