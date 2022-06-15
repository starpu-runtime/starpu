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

static __global__ void _increment_hip_codelet(unsigned *val)
{
	val[0]++;
}

extern "C" void increment_hip(void *descr[], void *cl_arg)
{
	unsigned *val = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

        hipLaunchKernelGGL(_increment_hip_codelet, 1, 1, 0, starpu_hip_get_local_stream(), val);
	hipError_t status = hipGetLastError();
	if (status != hipSuccess) STARPU_HIP_REPORT_ERROR(status);
}

extern "C" void redux_hip_kernel(void *descr[], void *arg)
{
	(void)arg;
	hipError_t ret;

	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);
	unsigned *src = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[1]);

	unsigned host_dst, host_src;

	/* This is a dummy technique of course */
	/* TODO: make this a kernel */
	ret = hipMemcpyAsync(&host_src, src, sizeof(unsigned), hipMemcpyDeviceToHost, starpu_hip_get_local_stream());
	if (STARPU_UNLIKELY(ret)) STARPU_HIP_REPORT_ERROR(ret);
	ret = hipMemcpyAsync(&host_dst, dst, sizeof(unsigned), hipMemcpyDeviceToHost, starpu_hip_get_local_stream());
	if (STARPU_UNLIKELY(ret)) STARPU_HIP_REPORT_ERROR(ret);
	ret = hipStreamSynchronize(starpu_hip_get_local_stream());
	if (STARPU_UNLIKELY(ret)) STARPU_HIP_REPORT_ERROR(ret);

	host_dst += host_src;

	ret = hipMemcpyAsync(dst, &host_dst, sizeof(unsigned), hipMemcpyHostToDevice, starpu_hip_get_local_stream());
	if (STARPU_UNLIKELY(ret)) STARPU_HIP_REPORT_ERROR(ret);
}

extern "C" void neutral_hip_kernel(void *descr[], void *arg)
{
	(void)arg;
	hipError_t ret;

	STARPU_SKIP_IF_VALGRIND;

	unsigned *dst = (unsigned *)STARPU_VARIABLE_GET_PTR(descr[0]);

	/* This is a dummy technique of course */
	unsigned host_dst = 0;
	ret = hipMemcpyAsync(dst, &host_dst, sizeof(unsigned), hipMemcpyHostToDevice, starpu_hip_get_local_stream());
	if (STARPU_UNLIKELY(ret)) STARPU_HIP_REPORT_ERROR(ret);
}
