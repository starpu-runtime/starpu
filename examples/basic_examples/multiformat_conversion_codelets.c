/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "multiformat_types.h"

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

#ifdef STARPU_USE_CUDA
void cuda_to_cpu(void *buffers[], void *arg)
{
	(void)arg;
	struct struct_of_arrays *src = STARPU_MULTIFORMAT_GET_CUDA_PTR(buffers[0]);
	struct point *dst = STARPU_MULTIFORMAT_GET_CPU_PTR(buffers[0]);
	int n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);
	int i;
	for (i = 0; i < n; i++)
	{
		dst[i].x = src->x[i];
		dst[i].y = src->y[i];
	}
}

extern void cpu_to_cuda_cuda_func(void *buffers[], void *args);

struct starpu_codelet cpu_to_cuda_cl =
{
	.cuda_funcs = {cpu_to_cuda_cuda_func},
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codelet_cpu_to_cuda"
};

struct starpu_codelet cuda_to_cpu_cl =
{
	.cpu_funcs = {cuda_to_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
	.name = "codelet_cude_to_cpu"
};
#endif

#ifdef STARPU_USE_OPENCL
void opencl_to_cpu(void *buffers[], void *arg)
{
	(void)arg;
	FPRINTF(stderr, "User Entering %s\n", __starpu_func__);
	struct struct_of_arrays *src = STARPU_MULTIFORMAT_GET_OPENCL_PTR(buffers[0]);
	struct point *dst = STARPU_MULTIFORMAT_GET_CPU_PTR(buffers[0]);
	int n = STARPU_MULTIFORMAT_GET_NX(buffers[0]);
	int i;
	for (i = 0; i < n; i++)
	{
		dst[i].x = src->x[i];
		dst[i].y = src->y[i];
	}
}

extern void cpu_to_opencl_opencl_func(void *buffers[], void *args);
struct starpu_codelet cpu_to_opencl_cl =
{
	.opencl_funcs = {cpu_to_opencl_opencl_func},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
	.nbuffers = 1,
	.modes = {STARPU_RW},
};

struct starpu_codelet opencl_to_cpu_cl =
{
	.cpu_funcs = {opencl_to_cpu},
	.nbuffers = 1,
	.modes = {STARPU_RW},
};
#endif
