/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012 INRIA
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
	unsigned int n = CUSTOM_GET_NX(buffers[0]);
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
	.where = STARPU_CUDA,
	.cuda_funcs = {cpu_to_cuda_cuda_func, NULL},
	.modes = { STARPU_RW },
	.nbuffers = 1,
	.name = "codelet_cpu_to_cuda"
};

struct starpu_codelet cuda_to_cpu_cl =
{
	.where = STARPU_CPU,
	.cpu_funcs = {cuda_to_cpu, NULL},
	.modes = { STARPU_RW },
	.nbuffers = 1,
	.name = "codelet_cuda_to_cpu"
};
#endif
