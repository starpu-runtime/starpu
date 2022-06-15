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

#ifdef STARPU_USE_CUDA
extern void increment_cuda(void *descr[], void *_args);
#endif
#ifdef STARPU_USE_HIP
extern void increment_hip(void *descr[], void *_args);
#endif
#ifdef STARPU_USE_OPENCL
extern void increment_opencl(void *buffers[], void *args);
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

#ifdef STARPU_USE_OPENCL
struct starpu_opencl_program opencl_program;
#endif

void increment_load_opencl()
{
#ifdef STARPU_USE_OPENCL
	int ret = starpu_opencl_load_opencl_from_file("tests/variable/increment_opencl_kernel.cl", &opencl_program, NULL);
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_load_opencl_from_file");
#endif
}

void increment_unload_opencl()
{
#ifdef STARPU_USE_OPENCL
        int ret = starpu_opencl_unload_opencl(&opencl_program);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_opencl_unload_opencl");
#endif
}


