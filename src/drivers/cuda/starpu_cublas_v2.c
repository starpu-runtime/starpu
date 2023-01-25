/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu_cuda.h>
#include <common/config.h>
#include <core/workers.h>

#ifdef STARPU_USE_CUDA
#include <drivers/cuda/driver_cuda.h>
#include <cublas_v2.h>
#include <starpu_cublas_v2.h>

//#ifdef CUBLAS_H_
//#error oops
//#endif

static cublasHandle_t cublas_handles[STARPU_NMAXWORKERS];
static cublasHandle_t main_handle;

void _starpu_init_cublas_v2_func(void)
{
	cublasCreate(&cublas_handles[starpu_worker_get_id_check()]);
	cublasSetStream(cublas_handles[starpu_worker_get_id_check()], starpu_cuda_get_local_stream());
}
void _starpu_shutdown_cublas_v2_func(void)
{
	cublasDestroy(cublas_handles[starpu_worker_get_id_check()]);
}

void _starpu_cublas_v2_init(void)
{
	if (cublasCreate(&main_handle) != CUBLAS_STATUS_SUCCESS)
		main_handle = NULL;
}

void _starpu_cublas_v2_shutdown(void)
{
	if (main_handle)
		cublasDestroy(main_handle);
}

cublasHandle_t starpu_cublas_get_local_handle(void)
{
	int workerid = starpu_worker_get_id();
	if (workerid >= 0)
		return cublas_handles[workerid];
	else
		return main_handle;
}
#endif
