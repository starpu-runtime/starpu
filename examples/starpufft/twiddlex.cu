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

extern "C" __global__ void STARPUFFT(cuda_1d_twiddle)(_cuComplex * out, _cuComplex * roots, unsigned n, unsigned i)
{
	unsigned j;
	unsigned start = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned numthreads = blockDim.x * gridDim.x;
	unsigned end = n;

	for (j = start; j < end; j += numthreads)
		out[j] = _cuCmul(out[j], roots[i*j]);
	return;
}

extern "C" void STARPUFFT(cuda_1d_twiddle_host)(_cuComplex *out, _cuComplex *roots, unsigned n, unsigned i)
{
	unsigned threads_per_block = 128;

	if (n < threads_per_block) {
		dim3 dimGrid(n);
		STARPUFFT(cuda_1d_twiddle) <<<dimGrid, 1>>> (out, roots, n, i);
	} else {
		dim3 dimGrid(n / threads_per_block);
		dim3 dimBlock(threads_per_block);
		STARPUFFT(cuda_1d_twiddle) <<<dimGrid, dimBlock>>> (out, roots, n, i);
	}
}

extern "C" __global__ void STARPUFFT(cuda_2d_twiddle)(_cuComplex * out, _cuComplex * roots0, _cuComplex * roots1, unsigned n2, unsigned m2, unsigned i, unsigned j)
{
	unsigned k, l;
	unsigned startx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned starty = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned numthreadsx = blockDim.x * gridDim.x;
	unsigned numthreadsy = blockDim.y * gridDim.y;
	unsigned endx = n2;
	unsigned endy = m2;

	for (k = startx; k < endx ; k += numthreadsx)
		for (l = starty; l < endy ; l += numthreadsy)
			out[k*m2 + l] = _cuCmul(_cuCmul(out[k*m2 + l], roots0[i*k]), roots1[j*l]);
	return;
}

extern "C" void STARPUFFT(cuda_2d_twiddle_host)(_cuComplex *out, _cuComplex *roots0, _cuComplex *roots1, unsigned n2, unsigned m2, unsigned i, unsigned j)
{
	unsigned threads_per_dim = 16;
	if (n2 < threads_per_dim) {
		if (m2 < threads_per_dim) {
			dim3 dimGrid(n2, m2);
			STARPUFFT(cuda_2d_twiddle) <<<dimGrid, 1>>> (out, roots0, roots1, n2, m2, i, j);
		} else {
			dim3 dimGrid(1, m2 / threads_per_dim);
			dim3 dimBlock(n2, threads_per_dim);
			STARPUFFT(cuda_2d_twiddle) <<<dimGrid, dimBlock>>> (out, roots0, roots1, n2, m2, i, j);
		}
	} else { 
		if (m2 < threads_per_dim) {
			dim3 dimGrid(n2 / threads_per_dim, 1);
			dim3 dimBlock(threads_per_dim, m2);
			STARPUFFT(cuda_2d_twiddle) <<<dimGrid, dimBlock>>> (out, roots0, roots1, n2, m2, i, j);
		} else {
			dim3 dimGrid(n2 / threads_per_dim, m2 / threads_per_dim);
			dim3 dimBlock(threads_per_dim, threads_per_dim);
			STARPUFFT(cuda_2d_twiddle) <<<dimGrid, dimBlock>>> (out, roots0, roots1, n2, m2, i, j);
		}
	}
}
