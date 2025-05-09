/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

static __global__ void _increment_hip(unsigned *val)
{
	val[0]++;
}

extern "C" void increment_hip(void *descr[], void *cl_arg)
{
	(void)cl_arg;
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

        hipLaunchKernelGGL(_increment_hip, 1, 1, 0, starpu_hip_get_local_stream(), val);
	hipError_t status = hipGetLastError();
	if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}

static __global__ void _redux_hip(unsigned *dst, unsigned *src)
{
	dst[0] += src[0];
}

extern "C" void redux_hip(void *descr[], void *cl_arg)
{
	(void)cl_arg;

	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	hipLaunchKernelGGL(_redux_hip, 1, 1, 0, starpu_hip_get_local_stream(), dst, src);
	hipError_t status = hipGetLastError();
	if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}

static __global__ void _neutral_hip(unsigned *dst)
{
	dst[0] = 0;
}

extern "C" void neutral_hip(void *descr[], void *cl_arg)
{
	(void)cl_arg;

	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	hipLaunchKernelGGL(_neutral_hip, 1, 1, 0, starpu_hip_get_local_stream(), dst);
	hipError_t status = hipGetLastError();
	if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}
