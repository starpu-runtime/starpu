/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include "../helper.h"
#include "increment.h"

#ifdef STARPU_USE_CUDA
extern void increment_cuda(void *descr[], void *_args);
extern void redux_cuda(void *descr[], void *arg);
extern void neutral_cuda(void *descr[], void *arg);
#endif
#ifdef STARPU_USE_HIP
extern void increment_hip(void *descr[], void *_args);
extern void redux_hip(void *descr[], void *arg);
extern void neutral_hip(void *descr[], void *arg);
#endif
#ifdef STARPU_USE_OPENCL
extern void increment_opencl(void *buffers[], void *args);
extern void redux_opencl(void *descr[], void *arg);
extern void neutral_opencl(void *descr[], void *arg);
#endif

void increment_cpu(void *descr[], void *arg)
{
	(void)arg;
	unsigned *tokenptr = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	(*tokenptr)++;
}

struct starpu_codelet increment_cl =
{
	.modes = {STARPU_RW},
	.cpu_funcs = {increment_cpu},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {increment_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
	.hip_funcs = {increment_hip},
	.hip_flags = {STARPU_HIP_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {increment_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"increment_cpu"},
	.nbuffers = 1
};

struct starpu_codelet increment_redux_cl =
{
	.modes = {STARPU_REDUX},
	.cpu_funcs = {increment_cpu},
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {increment_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
	.hip_funcs = {increment_hip},
	.hip_flags = {STARPU_HIP_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {increment_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs_name = {"increment_cpu"},
	.nbuffers = 1,
};

void redux_cpu(void *descr[], void *arg)
{
	(void)arg;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);
	*dst = *dst + *src;
}

struct starpu_codelet redux_cl =
{
	.modes = {STARPU_RW|STARPU_COMMUTE, STARPU_R},
	.nbuffers = 2,
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {redux_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
	.hip_funcs = {redux_hip},
	.hip_flags = {STARPU_HIP_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {redux_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs = {redux_cpu},
	.cpu_funcs_name = {"redux_cpu"},
};

void neutral_cpu(void *descr[], void *arg)
{
	(void)arg;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*dst = 0;
}

struct starpu_codelet neutral_cl =
{
#ifdef STARPU_USE_CUDA
	.cuda_funcs = {neutral_cuda},
	.cuda_flags = {STARPU_CUDA_ASYNC},
#endif
#ifdef STARPU_USE_HIP
	.hip_funcs = {neutral_hip},
	.hip_flags = {STARPU_HIP_ASYNC},
#endif
#ifdef STARPU_USE_OPENCL
	.opencl_funcs = {neutral_opencl},
	.opencl_flags = {STARPU_OPENCL_ASYNC},
#endif
	.cpu_funcs = {neutral_cpu},
	.cpu_funcs_name = {"neutral_cpu"},
	.modes = {STARPU_W},
	.nbuffers = 1
};

#ifndef STARPU_USE_OPENCL
void increment_load_opencl()
{
}
void increment_unload_opencl()
{
}
#endif
