/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/* Trivial dot reduction CUDA kernel */

#include <starpu.h>

#define DOT_TYPE double

static __global__ void cuda_redux(DOT_TYPE *dota, DOT_TYPE *dotb)
{
	*dota = *dota + *dotb;
	return;
}

extern "C" void redux_cuda_func(void *descr[], void *_args)
{
	(void)_args;
	DOT_TYPE *dota = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[0]);
	DOT_TYPE *dotb = (DOT_TYPE *)STARPU_VARIABLE_GET_PTR(descr[1]);

	cuda_redux<<<1,1, 0, starpu_cuda_get_local_stream()>>>(dota, dotb);
	cudaError_t status = cudaGetLastError();
	if (status != cudaSuccess) STARPU_CUDA_REPORT_ERROR(status);
}
