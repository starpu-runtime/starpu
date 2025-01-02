/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "custom_interface.h"
#include "custom_types.h"

#ifdef STARPU_USE_CUDA
void cuda_to_cpu(void *buffers[], void *arg)
{
	(void)arg;
	int n = CUSTOM_GET_NX(buffers[0]);
	float *x = (float*) CUSTOM_GET_X_PTR(buffers[0]);
	float *y = (float*) CUSTOM_GET_Y_PTR(buffers[0]);
	struct point *aop;
	aop = (struct point *) CUSTOM_GET_CPU_PTR(buffers[0]);

	int i;
	for (i = 0; i < n; i++)
	{
		aop[i].x = x[i];
		aop[i].y = y[i];
	}
	return;
}

extern void cpu_to_cuda_cuda_func(void *buffers[], void *args);
struct starpu_codelet cpu_to_cuda_cl =
{
	.cuda_funcs = {cpu_to_cuda_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.modes = { STARPU_RW },
	.nbuffers = 1,
	.name = "codelet_cpu_to_cuda"
};

struct starpu_codelet cuda_to_cpu_cl =
{
	.cpu_funcs = {cuda_to_cpu},
	.modes = { STARPU_RW },
	.nbuffers = 1,
	.name = "codelet_cuda_to_cpu"
};
#endif


#ifdef STARPU_USE_OPENCL
void opencl_to_cpu_cpu_func(void *buffers[], void *arg)
{
	(void)arg;
	int n = CUSTOM_GET_NX(buffers[0]);
	float *x = (float *) CUSTOM_GET_OPENCL_X_PTR(buffers[0]);
	struct point *aop;
	aop = (struct point *) CUSTOM_GET_CPU_PTR(buffers[0]);

	int i;
	for (i = 0; i < n; i++)
	{
		aop[i].x = x[i];
		aop[i].y = x[i+n];
	}
}

extern void cpu_to_opencl_opencl_func(void *buffers[], void *arg);

struct starpu_codelet cpu_to_opencl_cl =
{
	.opencl_funcs = { cpu_to_opencl_opencl_func },
	.opencl_flags = {STARPU_OPENCL_ASYNC},
	.modes = { STARPU_RW },
	.nbuffers = 1,
	.name = "codelet_cpu_to_opencl"
};

struct starpu_codelet opencl_to_cpu_cl =
{
	.cpu_funcs = { opencl_to_cpu_cpu_func },
	.modes = { STARPU_RW },
	.nbuffers = 1,
	.name = "codelet_opencl_to_cpu"
};
#endif /* !STARPU_USE_OPENCL */
